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

        # 定义每层的配置
        self.layer_configs = [
            {'in_channels': self.channels, 'out_channels': 16, 'kernel_size': 5, 'has_pool': True},  # 第一层
            {'in_channels': 16, 'out_channels': 32, 'kernel_size': 5, 'has_pool': True},  # 第二层
            {'in_channels': 32, 'out_channels': 64, 'kernel_size': 6, 'has_pool': True},  # 第三层
            {'in_channels': 64, 'out_channels': 128, 'kernel_size': 5, 'has_pool': False},  # 第四层
            {'in_channels': 128, 'out_channels': self.classes, 'kernel_size': 3, 'has_pool': False}  # 第五层
        ]

        print("  Creating CFMs...")
        t0 = time.time()
        # 为每一层创建对应的CFM
        self.cfms = nn.ModuleList()
        for config in self.layer_configs:
            # 计算每个CFM的输出维度：kernel_size * kernel_size * in_channels * out_channels + out_channels
            cfm_output_dim = (config['kernel_size'] * config['kernel_size'] * 
                            config['in_channels'] * config['out_channels'] + 
                            config['out_channels'])
            cfm_layer = cfm.CFM(self.cfm_input_dim, cfm_output_dim, cfm_type=self.cfm_type)
            self.cfms.append(cfm_layer)
        print(f"  CFMs creation took {time.time() - t0:.2f}s")

        # 创建常规卷积层作为备选
        print("  Creating regular conv layers as fallback...")
        t0 = time.time()
        _w_init = params.get('w_init', lambda x: nn.init.kaiming_normal_(x, nonlinearity='relu'))
        _b_init = params.get('b_init', lambda x: nn.init.constant_(x, 0.1))
        
        self.regular_convs = nn.ModuleList()
        for config in self.layer_configs:
            conv_block = _blocks.Conv2DBlock(
                shape=[config['kernel_size'], config['kernel_size'], 
                      config['in_channels'], config['out_channels']], 
                stride=1, padding='valid', 
                activation='relu' if config['out_channels'] != self.classes else None,
                max_pool=config['has_pool'],
                w_init=_w_init, 
                b_init=_b_init if config['out_channels'] != self.classes else nn.init.zeros_
            )
            self.regular_convs.append(conv_block)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.flatten = nn.Flatten()
        
        print(f"  Regular convs creation took {time.time() - t0:.2f}s")
        print(f"  Total Network initialization took {time.time() - t_start:.2f}s")

    def apply_cfm_conv_layer(self, x, cfm_input, layer_idx):
        # 获取当前层的配置
        config = self.layer_configs[layer_idx]
        batch_size, channels, height, width = x.shape
        
        # 重塑输入x为[1, B*C, H, W]
        x_reshaped = x.view(1, batch_size * channels, height, width)
        
        # 获取CFM输出的权重和偏置
        cfm_output = self.cfms[layer_idx](cfm_input)
        out_channels = config['out_channels']
        
        # 分离权重和偏置
        weights_size = config['kernel_size'] * config['kernel_size'] * channels * out_channels
        weights = cfm_output[:, :weights_size]
        bias = cfm_output[:, weights_size:]
        
        # 重塑权重为[B*out_channels, in_channels, kernel_size, kernel_size]
        weights = weights.reshape(batch_size * out_channels, channels, 
                                config['kernel_size'], config['kernel_size'])
        
        # 重塑偏置为[B*out_channels]
        bias = bias.reshape(-1)
        
        # 执行分组卷积
        x_conv = torch.nn.functional.conv2d(
            x_reshaped,
            weights,
            bias=bias,
            stride=1,
            padding=0,
            groups=batch_size
        )
        
        # 重塑输出为[B, out_channels, H, W]
        out_height = x_conv.shape[2]
        out_width = x_conv.shape[3]
        x_output = x_conv.view(batch_size, out_channels, out_height, out_width)
        
        # 应用激活函数和池化（如果需要）
        if layer_idx < len(self.layer_configs) - 1:  # 不是最后一层
            x_output = self.relu(x_output)
        if config['has_pool']:
            x_output = self.maxpool(x_output)
            
        return x_output

    def forward(self, x, cfm_input=None):
        if self.use_cfm and cfm_input is not None:
            # 使用CFM生成的卷积核
            for i in range(len(self.layer_configs)):
                if i == 3:  # 第四层之前添加dropout
                    x = self.dropout(x)
                x = self.apply_cfm_conv_layer(x, cfm_input, i)
        else:
            # 使用常规卷积层
            for i, conv in enumerate(self.regular_convs):
                if i == 3:  # 第四层之前添加dropout
                    x = self.dropout(x)
                x = conv(x)
                
        x = self.flatten(x)
        return x
