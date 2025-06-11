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

        _w_init = params.get('w_init', lambda x: nn.init.kaiming_normal_(x, nonlinearity='relu'))
        _b_init = params.get('b_init', lambda x: nn.init.constant_(x, 0.1))
        print(f"  Parameter setup took {time.time() - t0:.2f}s")

        print("  Creating CFM...")
        t0 = time.time()
        # 创建CFM实例
        self.cfm = cfm.CFM(self.cfm_input_dim, self.channels)
        print(f"  CFM creation took {time.time() - t0:.2f}s")

        print("  Creating first conv layer...")
        t0 = time.time()
        # 第一个卷积层单独定义，因为需要动态更新参数
        self.first_conv = _blocks.Conv2DBlock(
            shape=[5, 5, self.channels, 16], stride=1, padding='valid', activation='relu', max_pool=True,
            w_init=_w_init, b_init=_b_init
        )
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
                shape=[5, 5, 64, 128], stride=1, padding='valid', activation='relu',
                w_init=_w_init, b_init=_b_init
            ),
            nn.Dropout(p=self.dropout_rate),
            _blocks.Conv2DBlock(
                shape=[3, 3, 128, self.classes], stride=1, padding='valid',
                w_init=_w_init, b_init=nn.init.zeros_
            ),
            nn.Flatten()
        )
        print(f"  Remaining layers creation took {time.time() - t0:.2f}s")
        print(f"  Total Network initialization took {time.time() - t_start:.2f}s")

    def update_conv_weights(self, cfm_input):
        # 使用CFM生成新的卷积核参数
        new_weights = self.cfm(cfm_input)  # [B, 5, 5, 1, 16]
        
        # 检查batch size
        if len(new_weights.shape) == 5:  # 如果有batch维度
            # 对每个batch生成的权重取平均，或者只使用第一个
            new_weights = new_weights.mean(dim=0)  # [5, 5, 1, 16]
            
        # 调整维度顺序以匹配PyTorch的卷积权重格式 [out_channels, in_channels, height, width]
        new_weights = new_weights.permute(3, 2, 0, 1)  # [16, 1, 5, 5]
        
        # 更新第一个卷积层的权重
        conv_layer = self.first_conv._layer.conv
        conv_layer.weight.data = new_weights

    def forward(self, x, cfm_input=None):
        if self.use_cfm and cfm_input is not None:
            self.update_conv_weights(cfm_input)
        
        x = self.first_conv(x)
        x = self.remaining_layers(x)
        return x
