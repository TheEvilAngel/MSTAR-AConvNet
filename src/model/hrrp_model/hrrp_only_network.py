import torch.nn as nn
import torch
from .. import cfm

class HRRPOnlyNetwork(nn.Module):
    def __init__(self, **params):
        super(HRRPOnlyNetwork, self).__init__()
        
        self.cfm_input_dim = params.get('cfm_input_dim', 100)
        self.classes = params.get('classes', 10)
        self.cfm_type = params.get('cfm_type', 'hrrp_linear')
        
        # 创建CFM层
        self.cfm = cfm.CFM(self.cfm_input_dim, 512, cfm_type=self.cfm_type)
        
        # 创建分类层
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=params.get('dropout_rate', 0.5)),
            nn.Linear(256, self.classes)
        )
        
    def forward(self, hrrp_data):
        # 只处理HRRP数据
        x = self.cfm(hrrp_data)
        x = self.classifier(x)
        return x 