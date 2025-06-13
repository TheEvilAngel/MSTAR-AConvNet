import numpy as np

from skimage import io
import torch
import tqdm

import json
import glob
import os
from collections import defaultdict
import pdb

# import utils.common as common
project_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class Dataset(torch.utils.data.Dataset):

    def __init__(self, path, name='soc', data_type='sar', data_hrrp_type='hrrp_column_complex', is_train=False, use_cfm=False, column_group_size=100, transform=None):
        self.is_train = is_train
        self.use_cfm = use_cfm
        self.name = name
        self.data_type = data_type
        self.data_hrrp_type = data_hrrp_type
        self.column_group_size = column_group_size 

        self.images = []          # sar images
        self.hrrp_data = []      # hrrp data for cfm
        self.labels = []         # labels
        self.serial_number = []  # serial numbers
        self.angles = []         # azimuth angles for sar
        
        self.transform = transform
        
        # 建立HRRP数据的索引
        self.hrrp_index = None
        if self.use_cfm:
            self.hrrp_index = self._build_hrrp_index(path)
            
        self._load_data(path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        _image = self.images[idx]
        _label = self.labels[idx]

        if self.transform:
            _image = self.transform(_image)
        # TODO hrrp not transform

        if self.use_cfm:
            return _image, _label, self.serial_number[idx], self.hrrp_data[idx]
        return _image, _label, self.serial_number[idx]

    def _normalize_angle(self, angle):
        """标准化角度，使其在0-180度之间"""
        angle = float(angle)
        if angle > 180:
            angle -= 180
        return angle

    def _build_hrrp_index(self, path):
        """建立HRRP数据的索引，格式为：{class_id: {angle: file_path}}"""
        mode = 'train' if self.is_train else 'test'
        hrrp_files = glob.glob(os.path.join(
            project_root, path,
            f'{self.name}/{mode}/{self.data_hrrp_type}/{self.column_group_size}/*/*.npy'
        ))
        
        # 使用defaultdict避免重复检查字典键是否存在
        index = defaultdict(dict)
        
        print(f"Building HRRP index for {mode} set...")
        for hrrp_file in tqdm.tqdm(hrrp_files):
            json_file = hrrp_file.replace('.npy', '.json')
            if not os.path.exists(json_file):
                continue
                
            with open(json_file, 'r') as f:
                info = json.load(f)
            
            class_id = info['class_id']
            angle = self._normalize_angle(info['azimuth_angle'])
            index[class_id][angle] = hrrp_file
            
        return index

    def _find_closest_hrrp(self, target_label, target_angle):
        """使用索引快速找到最接近目标角度+30度的HRRP数据"""
        if target_label not in self.hrrp_index:
            return None
            
        target_angle = self._normalize_angle(target_angle)
        target_angle_plus_30 = self._normalize_angle(target_angle + 30)
        angles = list(self.hrrp_index[target_label].keys())
        
        if not angles:
            return None
            
        # 找到最接近目标角度+30的角度
        closest_angle = min(angles, key=lambda x: abs(x - target_angle_plus_30))
        return self.hrrp_index[target_label][closest_angle]

    def _load_data(self, path):
        mode = 'train' if self.is_train else 'test'
        
        # 加载SAR数据
        sar_image_list = glob.glob(os.path.join(project_root, path, f'{self.name}/{mode}/{self.data_type}/*/*.npy'))
        sar_label_list = glob.glob(os.path.join(project_root, path, f'{self.name}/{mode}/{self.data_type}/*/*.json'))
        sar_image_list = sorted(sar_image_list, key=os.path.basename)
        sar_label_list = sorted(sar_label_list, key=os.path.basename)

        for image_path, label_path in tqdm.tqdm(zip(sar_image_list, sar_label_list), desc=f'load {mode} data set'):
            # 加载SAR数据
            self.images.append(np.load(image_path))

            # 加载标签信息
            with open(label_path, mode='r', encoding='utf-8') as f:
                _label = json.load(f)

            self.labels.append(_label['class_id'])
            self.serial_number.append(_label['serial_number'])
            
            angle = self._normalize_angle(_label['azimuth_angle'])
            self.angles.append(angle)
            
            # 如果使用CFM，找到对应的HRRP数据
            if self.use_cfm:
                hrrp_file = self._find_closest_hrrp(_label['class_id'], angle)
                if hrrp_file is not None:
                    # 直接加载HRRP数据
                    hrrp_data = np.load(hrrp_file)
                    self.hrrp_data.append(torch.FloatTensor(hrrp_data))
                else:
                    # 如果找不到对应的HRRP数据，使用零向量
                    pdb.set_trace()
                    self.hrrp_data.append(torch.zeros(self.column_group_size))
