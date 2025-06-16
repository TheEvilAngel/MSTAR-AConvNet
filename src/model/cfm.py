import torch
import torch.nn as nn
from .hrrp_model.hrrp_linear import HRRPLinear
from .hrrp_model.hrrp_graph import HRRPGraphNet
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
        else:
            raise ValueError(f"Unknown CFM type: {cfm_type}")
            
    def forward(self, x, distance_matrix=None):
        if self.cfm_type == 'hrrp_linear':
            return self.model(x)
        elif self.cfm_type == 'hrrp_graph':
            if x.dim() == 2:
                x = x.unsqueeze(1)
            return self.model(x, self.distance_matrix.to(x.device)) 