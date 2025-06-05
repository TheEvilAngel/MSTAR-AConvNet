import torch
import os 
import pdb

os.environ['CUDA_VISIBLE_DEVICES'] = '7'

# 设置模型文件路径
path = "/home/chenzihong/doc/MSTAR-AConvNet/experiments/model/AConvNet-SOC-sar49-usephase/20250603_111012/model-200.pth"  # 请将此路径替换为您的实际模型文件路径
# path = "/home/chenzihong/doc/MSTAR-AConvNet/experiments/model/AConvNet-SOC-sar1-usephase/20250603_105846/model-200.pth"

# 加载模型参数
checkpoint = torch.load(path)

# 打印模型的所有键
print("\n模型字典中的键：")
for key in checkpoint.keys():
    print(f"- {key}")

pdb.set_trace()
# 如果是state_dict格式，打印每层的参数形状
if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint

print("\n各层参数的形状：")
for param_name, param in state_dict.items():
    print(f"{param_name}: {param.shape}")

# 打印参数统计信息
total_params = sum(p.numel() for p in state_dict.values())
print(f"\n总参数量：{total_params:,}") 

