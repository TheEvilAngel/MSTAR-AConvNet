import torch
import torch.nn as nn
from .hrrp_model.hrrp_linear import HRRPLinear
from .hrrp_model.hrrp_graph import HRRPGraphNet
from .hrrp_model.resnet_1D import ResNet18_1D, ResNet34_1D, ResNet50_1D, ResNet101_1D, ResNet152_1D
import pdb


class CFM(nn.Module):
    def __init__(self, input_dim, output_dim, cfm_type='hrrp_linear'):
        super(CFM, self).__init__()
        self.cfm_type = cfm_type
        
        if cfm_type == 'hrrp_linear':
            self.model = HRRPLinear(input_dim, output_dim)
        elif cfm_type == 'hrrp_graph':
            self.model = HRRPGraphNet(input_dim, output_dim)
            self.distance_matrix = self.model.generate_distance_matrix(input_dim)
        elif cfm_type == 'resnet18':
            self.model = ResNet18_1D(input_dim, output_dim)
        elif cfm_type == 'resnet34':
            self.model = ResNet34_1D(input_dim, output_dim)
        elif cfm_type == 'resnet50':
            self.model = ResNet50_1D(input_dim, output_dim)
        elif cfm_type == 'resnet101':
            self.model = ResNet101_1D(input_dim, output_dim)
        elif cfm_type == 'resnet152':
            self.model = ResNet152_1D(input_dim, output_dim)
        else:
            raise ValueError(f"Unknown CFM type: {cfm_type}")
            
    def forward(self, x, distance_matrix=None):
        if self.cfm_type == 'hrrp_linear':
            return self.model(x)
        elif self.cfm_type == 'hrrp_graph':
            if x.dim() == 2:
                x = x.unsqueeze(1)
            return self.model(x, self.distance_matrix.to(x.device)) 
        elif self.cfm_type == 'resnet18' or self.cfm_type == 'resnet34' or self.cfm_type == 'resnet50' or self.cfm_type == 'resnet101' or self.cfm_type == 'resnet152':
            if x.dim() == 2:
                x = x.unsqueeze(1)
            return self.model(x)