import os
from absl import logging
from absl import flags
from absl import app

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from datetime import datetime

from data import loader
from utils import common
from model.hrrp_model.hrrp_only_network import HRRPOnlyNetwork

flags.DEFINE_string('experiments_path', os.path.join(common.project_root, 'experiments'), help='')
flags.DEFINE_string('config_name', 'config/hrrp-only.json', help='')
FLAGS = flags.FLAGS

def load_hrrp_dataset(path, is_train, name, data_type, data_hrrp_type, batch_size, column_group_size):
    _dataset = loader.Dataset(
        path, 
        name=name,
        data_type=data_type,
        data_hrrp_type=data_hrrp_type,
        is_train=is_train,
        use_cfm=True,  # 必须为True以获取HRRP数据
        column_group_size=column_group_size
    )
    
    data_loader = torch.utils.data.DataLoader(
        _dataset, 
        batch_size=batch_size, 
        shuffle=is_train, 
        num_workers=4
    )
    return data_loader

class HRRPModel:
    def __init__(self, **params):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建网络
        self.net = HRRPOnlyNetwork(
            classes=params['num_classes'],  # 修改为与train.py一致的参数名
            cfm_input_dim=params['cfm_input_dim'],
            dropout_rate=params['dropout_rate'],
            cfm_type=params['cfm_type']
        ).to(self.device)
        
        # 创建优化器
        self.optimizer = optim.SGD(
            self.net.parameters(),
            lr=params['lr'],
            momentum=0.9,  # 设置默认动量
            weight_decay=params['weight_decay']
        )
        
        # 学习率调度器
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=params['lr_step'],
            gamma=params['lr_decay']
        )
        
        self.criterion = nn.CrossEntropyLoss()
    
    def optimize(self, hrrp_data, labels):
        self.optimizer.zero_grad()
        
        # 前向传播
        outputs = self.net(hrrp_data.to(self.device))
        loss = self.criterion(outputs, labels.to(self.device))
        
        # 反向传播
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def inference(self, hrrp_data):
        return self.net(hrrp_data.to(self.device))
    
    def save(self, path):
        torch.save(self.net.state_dict(), path)
    
    def load(self, path):
        self.net.load_state_dict(torch.load(path))

@torch.no_grad()
def validation(model, valid_set):
    num_data = 0
    corrects = 0

    # Test loop
    model.net.eval()
    _softmax = torch.nn.Softmax(dim=1)
    
    for _, labels, _, hrrp_data in tqdm(valid_set):
        predictions = model.inference(hrrp_data)
        predictions = _softmax(predictions)

        _, predictions = torch.max(predictions.data, 1)
        labels = labels.type(torch.LongTensor)
        num_data += labels.size(0)
        corrects += (predictions == labels.to(model.device)).sum().item()

    accuracy = 100 * corrects / num_data
    return accuracy

def run(epochs, dataset, classes, batch_size,
        lr, lr_step, lr_decay, weight_decay, dropout_rate,
        model_name, data_type, data_hrrp_type, train_type, 
        column_group_size, cfm_input_dim, cfm_type,
        experiments_path=None):

    datetime_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = os.path.join(experiments_path, f'model/{model_name}/{train_type}')
    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)

    history_path = os.path.join(experiments_path, 'history')
    if not os.path.exists(history_path):
        os.makedirs(history_path, exist_ok=True)

    train_set = load_hrrp_dataset('dataset', True, dataset, data_type, data_hrrp_type, batch_size, column_group_size)
    valid_set = load_hrrp_dataset('dataset', False, dataset, data_type, data_hrrp_type, batch_size, column_group_size)
    print(f'Train set size: {len(train_set.dataset)}')
    print(f'Validation set size: {len(valid_set.dataset)}')

    model = HRRPModel(
        num_classes=classes,
        dropout_rate=dropout_rate,
        lr=lr,
        lr_step=lr_step,
        lr_decay=lr_decay,
        weight_decay=weight_decay,
        cfm_input_dim=cfm_input_dim,
        cfm_type=cfm_type
    )

    history = {
        'loss': [],
        'accuracy': [],
        'best_accuracy': 0.0,
        'best_epoch': 0,
        'config': {
            'dataset': dataset,
            'num_classes': classes,
            'epochs': epochs,
            'batch_size': batch_size,
            'lr': lr,
            'lr_step': lr_step,
            'lr_decay': lr_decay,
            'weight_decay': weight_decay,
            'dropout_rate': dropout_rate,
            'model_name': model_name,
            'data_type': data_type,
            'train_type': train_type,
            'data_hrrp_type': data_hrrp_type,
            'column_group_size': column_group_size,
            'cfm_input_dim': cfm_input_dim,
            'cfm_type': cfm_type
        }
    }

    best_accuracy = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        _loss = []

        model.net.train()
        for _, labels, _, hrrp_data in tqdm(train_set):
            _loss.append(model.optimize(hrrp_data, labels))

        if model.lr_scheduler:
            current_lr = model.lr_scheduler.get_last_lr()[0]
            model.lr_scheduler.step()
            accuracy = validation(model, valid_set)
            logging.info(
                f'Epoch: {epoch + 1:03d}/{epochs:03d} | loss={np.mean(_loss):.4f} | lr={current_lr:.6f} | accuracy={accuracy:.2f}'
            )
        else:
            accuracy = validation(model, valid_set)
            logging.info(
                f'Epoch: {epoch + 1:03d}/{epochs:03d} | loss={np.mean(_loss):.4f} | accuracy={accuracy:.2f}'
            )

        history['loss'].append(np.mean(_loss))
        history['accuracy'].append(accuracy)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch + 1
            model.save(os.path.join(model_path, 'model-best.pth'))
            
    history['best_accuracy'] = best_accuracy
    history['best_epoch'] = best_epoch
    
    # 保存历史记录
    with open(os.path.join(history_path, f'history-{model_name}-{train_type}-{datetime_str}.json'), mode='w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=True, indent=2)

def main(_):
    logging.info('Start')
    experiments_path = FLAGS.experiments_path
    config_name = FLAGS.config_name

    config = common.load_config(os.path.join(experiments_path, config_name))

    # 从配置文件中读取GPU设置
    gpu_id = config.get('gpu_id', '0')  # 默认使用GPU 0
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    logging.info(f'Using GPU: {gpu_id}')

    dataset = config['dataset']
    classes = config['num_classes']
    epochs = config['epochs']
    batch_size = config['batch_size']

    lr = config['lr']
    lr_step = config['lr_step']
    lr_decay = config['lr_decay']

    weight_decay = config['weight_decay']
    dropout_rate = config['dropout_rate']

    model_name = config['model_name']

    data_type = config['data_type']
    data_hrrp_type = config['data_hrrp_type']
    train_type = config['train_type']
    
    cfm_input_dim = config['cfm_input_dim']
    cfm_type = config['cfm_type']
    column_group_size = config['column_group_size']
    
    run(epochs, dataset, classes, batch_size,
        lr, lr_step, lr_decay, weight_decay, dropout_rate,
        model_name, data_type, data_hrrp_type, train_type, 
        column_group_size, cfm_input_dim, cfm_type,
        experiments_path)

    logging.info('Finish')

if __name__ == '__main__':
    app.run(main) 