import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

from absl import logging
from absl import flags
from absl import app

from tqdm import tqdm

from torch.utils import tensorboard

import torchvision
import torch

import numpy as np

import json

from datetime import datetime

from data import preprocess
from data import loader
from utils import common
import model

import pdb

flags.DEFINE_string('experiments_path', os.path.join(common.project_root, 'experiments'), help='')
flags.DEFINE_string('config_name', 'config/AConvNet-SOC-sar.json', help='')
FLAGS = flags.FLAGS


common.set_random_seed(12321)


def load_dataset(path, is_train, use_cfm, name, data_type, data_hrrp_type, batch_size, column_group_size):
    transform = [preprocess.CenterCrop(88), torchvision.transforms.ToTensor()]
    if is_train:
        transform = [preprocess.RandomCrop(88), torchvision.transforms.ToTensor()]
    _dataset = loader.Dataset(
        path, name=name, data_type=data_type, data_hrrp_type=data_hrrp_type, is_train=is_train, use_cfm=use_cfm, column_group_size=column_group_size,
        transform=torchvision.transforms.Compose(transform)
    )
    # TODO 看是不是对应的读取
    data_loader = torch.utils.data.DataLoader(
        _dataset, batch_size=batch_size, shuffle=is_train, num_workers=4
    )
    return data_loader


@torch.no_grad()
def validation(m, ds):
    num_data = 0
    corrects = 0

    # Test loop
    m.net.eval()
    _softmax = torch.nn.Softmax(dim=1)
    for i, data in enumerate(tqdm(ds)):        
        if m.net.use_cfm:
            images, labels, _, hrrp_data = data
            predictions = m.inference(images, hrrp_data)
        else:
            images, labels, _ = data
            predictions = m.inference(images)
            
        predictions = _softmax(predictions)

        _, predictions = torch.max(predictions.data, 1)
        labels = labels.type(torch.LongTensor)
        num_data += labels.size(0)
        corrects += (predictions == labels.to(m.device)).sum().item()

    accuracy = 100 * corrects / num_data
    return accuracy


def run(epochs, dataset, classes, channels, batch_size,
        lr, lr_step, lr_decay, weight_decay, dropout_rate,
        model_name, data_type, data_hrrp_type, train_type, column_group_size, cfm_input_dim, use_cfm, experiments_path=None):
    train_set = load_dataset('dataset', True, use_cfm, dataset, data_type, data_hrrp_type, batch_size, column_group_size)
    valid_set = load_dataset('dataset', False, use_cfm, dataset, data_type, data_hrrp_type, batch_size, column_group_size)
    print(f'Train set size: {len(train_set.dataset)}')
    print(f'Validation set size: {len(valid_set.dataset)}')

    m = model.Model(
        classes=classes, dropout_rate=dropout_rate, channels=channels,
        lr=lr, lr_step=lr_step, lr_decay=lr_decay,
        weight_decay=weight_decay, cfm_input_dim=cfm_input_dim, use_cfm=use_cfm
    )

    datetime_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = os.path.join(experiments_path, f'model/{model_name}/{train_type}/{datetime_str}')
    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)

    history_path = os.path.join(experiments_path, 'history')
    if not os.path.exists(history_path):
        os.makedirs(history_path, exist_ok=True)

    history = {
        'loss': [],
        'accuracy': []
    }

    # 在保存history之前，添加配置信息
    history['config'] = {
        'dataset': dataset,
        'num_classes': classes,
        'channels': channels,
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
        "data_hrrp_type": "hrrp_column_complex",
        'column_group_size': column_group_size,
        'cfm_input_dim': cfm_input_dim,
        'use_cfm': use_cfm
    }

    best_accuray = 0.0
    # pdb.set_trace()
    for epoch in range(epochs):
        _loss = []

        m.net.train()
        for i, data in enumerate(tqdm(train_set)):
            if use_cfm:
                images, labels, _, hrrp_data = data
                _loss.append(m.optimize(images, labels, hrrp_data))
            else:
                images, labels, _ = data
                _loss.append(m.optimize(images, labels))

        if m.lr_scheduler:
            lr = m.lr_scheduler.get_last_lr()[0]
            m.lr_scheduler.step()

        accuracy = validation(m, valid_set)

        logging.info(
            f'Epoch: {epoch + 1:03d}/{epochs:03d} | loss={np.mean(_loss):.4f} | lr={lr} | accuracy={accuracy:.2f}'
        )

        history['loss'].append(np.mean(_loss))
        history['accuracy'].append(accuracy)
        if accuracy > best_accuray:
            best_accuray = accuracy
            m.save(os.path.join(model_path, f'model-best.pth'))

        if experiments_path:
            m.save(os.path.join(model_path, f'model-{epoch + 1:03d}.pth'))

    history['best_accuary'] = best_accuray
    
    sorted_accuracies = sorted(history['accuracy'], reverse=True)
    top10_avg_accuracy = sum(sorted_accuracies[:10]) / min(10, len(sorted_accuracies))
    history['top10_avg_acc'] = top10_avg_accuracy
    
    with open(os.path.join(history_path, f'history-{model_name}-{train_type}-{datetime_str}.json'), mode='w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=True, indent=2)
        


def main(_):
    logging.info('Start')
    experiments_path = FLAGS.experiments_path
    config_name = FLAGS.config_name

    config = common.load_config(os.path.join(experiments_path, config_name))

    dataset = config['dataset']
    classes = config['num_classes']
    channels = config['channels']
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
    use_cfm = config['use_cfm']
    column_group_size = config['column_group_size']
    
    run(epochs, dataset, classes, channels, batch_size,
        lr, lr_step, lr_decay, weight_decay, dropout_rate,
        model_name, data_type, data_hrrp_type, train_type, column_group_size, cfm_input_dim, use_cfm, experiments_path)

    logging.info('Finish')


if __name__ == '__main__':
    app.run(main)
