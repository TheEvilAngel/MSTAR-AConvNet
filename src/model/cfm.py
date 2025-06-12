import torch
import torch.nn as nn


class CFM(nn.Module):
    def __init__(self, input_dim, channels=1):
        super(CFM, self).__init__()
        # 计算目标输出维度：5*5*channels*16
        self.output_dim = 5 * 5 * channels * 16
        
        # 三层线性层
        self.linear1 = nn.Linear(input_dim, 256)
        self.linear2 = nn.Linear(256, 512)
        self.linear3 = nn.Linear(512, self.output_dim)
        
        # 激活函数
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # 确保输入是二维的 [batch_size, input_dim]
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        
        batch_size = x.shape[0]
        # 重塑输出为卷积核的形状 [batch_size, 5, 5, channels, 16]
        x = x.view(batch_size, -1)
        return x 