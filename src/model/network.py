import torch.nn as nn
import torch
import time
import pdb

from . import _blocks
from . import cfm


class Network(nn.Module):

    def __init__(self, **params):
        print("  Network.__init__ starting...")
        t_start = time.time()
        
        super(Network, self).__init__()
        
        print("  Setting up network parameters...")
        t0 = time.time()
        self.dropout_rate = params.get('dropout_rate', 0.5)
        self.classes = params.get('classes', 10)
        self.channels = params.get('channels', 1)
        self.cfm_input_dim = params.get('cfm_input_dim', 100)  # CFM的输入维度
        self.use_cfm = params.get('use_cfm', True)  # 控制是否使用CFM
        self.cfm_type = params.get('cfm_type', 'hrrp_linear')
        self.manifold_input_channels = params.get('manifold_input_channels', 64)
        self.manifold_output_channels = params.get('manifold_output_channels', 128)
        self.kernel_size = params.get('kernel_size', 3)

        _w_init = params.get('w_init', lambda x: nn.init.kaiming_normal_(x, nonlinearity='relu'))
        _b_init = params.get('b_init', lambda x: nn.init.constant_(x, 0.1))
        print(f"  Parameter setup took {time.time() - t0:.2f}s")

        print("  Creating CFM...")
        t0 = time.time()
        # 只在use_cfm为True时创建CFM实例
        self.cfm = cfm.CFM(self.cfm_input_dim, self.manifold_input_channels, cfm_type=self.cfm_type) if self.use_cfm else None
        self.weight_transform = nn.Linear(self.manifold_input_channels, self.kernel_size*self.kernel_size*self.manifold_input_channels*self.manifold_output_channels)
        self.bias_transform = nn.Linear(self.manifold_input_channels, self.manifold_output_channels)
        self.class_dropout = nn.Dropout(p=self.dropout_rate)
        self.class_fc = nn.Linear(self.manifold_output_channels, self.classes)
        print(f"  CFM creation took {time.time() - t0:.2f}s")

        print("  Creating first conv layer...")
        t0 = time.time()
        # 创建两个并行的第一层：一个用于CFM，一个用于普通Conv2D
        self.first_conv_normal = _blocks.Conv2DBlock(
            shape=[5, 5, self.channels, 16], stride=1, padding='valid', activation='relu', max_pool=True,
            w_init=_w_init, b_init=_b_init
        )
        self.first_maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.first_relu = nn.ReLU()
        print(f"  First conv creation took {time.time() - t0:.2f}s")

        print("  Creating remaining layers...")
        t0 = time.time()
        # 其余层保持不变
        self.remaining_layers = nn.Sequential(
            _blocks.Conv2DBlock(
                shape=[5, 5, 16, 32], stride=1, padding='valid', activation='relu', max_pool=True,
                w_init=_w_init, b_init=_b_init
            ),
            _blocks.Conv2DBlock(
                shape=[6, 6, 32, 64], stride=1, padding='valid', activation='relu', max_pool=True,
                w_init=_w_init, b_init=_b_init
            ),
            _blocks.Conv2DBlock(
                shape=[5, 5, 64, self.manifold_input_channels], stride=1, padding='valid', activation='relu',
                w_init=_w_init, b_init=_b_init
            ),
            # nn.Dropout(p=self.dropout_rate),
            # _blocks.Conv2DBlock(
            #     shape=[3, 3, 128, self.manifold_input_channels], stride=1, padding='valid',
            #     w_init=_w_init, b_init=nn.init.zeros_
            # ),
            # nn.Flatten()
        )
        print(f"  Remaining layers creation took {time.time() - t0:.2f}s")
        print(f"  Total Network initialization took {time.time() - t_start:.2f}s")

    def apply_cfm_first_conv(self, x, cfm_input):
        # 获取输入维度
        batch_size, channels, height, width = x.shape  # [B, C, 88, 88]
        
        # 1. 重塑输入x为[1, B*C, 88, 88]
        x_reshaped = x.view(1, batch_size * channels, height, width)
        
        # 2. 处理CFM输出的权重和偏置
        cfm_output = self.cfm(cfm_input)  # [B, 5*5*C*16 + 16]
        # 分离权重和偏置
        weights = cfm_output[:, :-16]  # [B, 5*5*C*16]
        bias = cfm_output[:, -16:]     # [B, 16]
        
        # 直接重塑权重为[B*16, C, 5, 5]
        weights = weights.reshape(batch_size * 16, channels, 5, 5)
        
        # 重塑偏置为[B*16]
        bias = bias.reshape(-1)  # [B*16]
        
        # 3. 执行分组卷积
        x_conv = torch.nn.functional.conv2d(
            x_reshaped,           # [1, B*C, 88, 88]
            weights,              # [B*16, C, 5, 5]
            bias=bias,            # [B*16]
            stride=1,
            padding=0,
            groups=batch_size     # 分组数等于batch_size
        )
        
        # 4. 重塑输出为[B, 16, H, W]
        out_height = x_conv.shape[2]  # 计算卷积后的高度
        out_width = x_conv.shape[3]   # 计算卷积后的宽度
        x_output = x_conv.view(batch_size, 16, out_height, out_width)
        
        # 应用激活函数和池化
        x_output = self.first_relu(x_output)
        x_output = self.first_maxpool(x_output)
        
        return x_output

    def apply_cfm_conv(self, x, cfm_input):
        # 获取输入维度
        # pdb.set_trace()
        batch_size, channels, height, width = x.shape  # [B, C, h, w]
        assert channels == self.manifold_input_channels, "Input channels must match manifold input channels"
        
        # 1. 重塑输入x为[1, B*C, h, w]
        x_reshaped = x.view(1, batch_size * channels, height, width)
        
        # 2. 处理CFM输出的权重和偏置
        cfm_output = self.cfm(cfm_input)  # [B, 5*5*C*16 + 16]
        # 分离权重和偏置
        weights = self.weight_transform(cfm_output)
        bias = self.bias_transform(cfm_output)
        
        # 直接重塑权重为[B*16, C, 5, 5]
        weights = weights.view(batch_size * self.manifold_output_channels, channels, self.kernel_size, self.kernel_size)
        
        # 重塑偏置为[B*16]
        bias = bias.view(batch_size * self.manifold_output_channels)  # [B*16]
        
        # 3. 执行分组卷积
        x_conv = torch.nn.functional.conv2d(
            x_reshaped,           # [1, B*C, 88, 88]
            weights,              # [B*16, C, 5, 5]
            bias=bias,            # [B*16]
            stride=1,
            padding=0,
            groups=batch_size     # 分组数等于batch_size
        )
        
        # 4. 重塑输出为[B, 16, H, W]
        out_height = x_conv.shape[2]  # 计算卷积后的高度
        out_width = x_conv.shape[3]   # 计算卷积后的宽度
        x_output = x_conv.view(batch_size, self.manifold_output_channels, out_height, out_width)
        
        # pdb.set_trace()
        x_output = x_output.view(batch_size, -1)
        x_output = self.class_dropout(x_output)
        x_output = self.class_fc(x_output)
        return x_output
    
    def forward(self, x, cfm_input=None):
        # 根据use_cfm选择使用哪个路径
        if self.use_cfm and cfm_input is not None:
            # x = self.apply_cfm_first_conv(x, cfm_input)
            x = self.first_conv_normal(x)
            x = self.remaining_layers(x)
            x = self.apply_cfm_conv(x, cfm_input)
        else:
            x = self.first_conv_normal(x)  
            x = self.remaining_layers(x)
        return x
