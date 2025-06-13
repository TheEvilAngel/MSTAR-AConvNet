import os
# 移除硬编码的GPU设置
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'

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


def save_checkpoint(checkpoint_path, run_idx, epoch, model, optimizer, scheduler, all_runs_history):
    checkpoint = {
        'run_idx': run_idx,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'all_runs_history': all_runs_history,
        'random_state': np.random.get_state(),
        'torch_random_state': torch.get_rng_state(),
        'cuda_random_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    }
    
    # 先保存到临时文件
    temp_checkpoint_path = checkpoint_path + '.tmp'
    try:
        torch.save(checkpoint, temp_checkpoint_path)
        # 如果保存成功，则替换原文件
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        os.rename(temp_checkpoint_path, checkpoint_path)
        logging.info(f'成功保存检查点：run {run_idx + 1}, epoch {epoch + 1}')
    except Exception as e:
        logging.error(f'保存检查点失败：{str(e)}')
        if os.path.exists(temp_checkpoint_path):
            os.remove(temp_checkpoint_path)
        raise e

def load_checkpoint(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        return None
        
    try:
        checkpoint = torch.load(checkpoint_path)
        # 恢复随机数状态
        np.random.set_state(checkpoint['random_state'])
        torch.set_rng_state(checkpoint['torch_random_state'])
        if checkpoint['cuda_random_state'] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(checkpoint['cuda_random_state'])
        logging.info(f'成功加载检查点：run {checkpoint["run_idx"] + 1}, epoch {checkpoint["epoch"] + 1}')
        return checkpoint
    except Exception as e:
        logging.error(f'加载检查点失败：{str(e)}')
        # 如果检查点文件损坏，将其重命名为.corrupted
        corrupted_path = checkpoint_path + '.corrupted'
        if os.path.exists(corrupted_path):
            os.remove(corrupted_path)
        os.rename(checkpoint_path, corrupted_path)
        logging.warning(f'已将损坏的检查点文件重命名为：{corrupted_path}')
        return None

def run(epochs, dataset, classes, channels, batch_size,
        lr, lr_step, lr_decay, weight_decay, dropout_rate,
        model_name, data_type, data_hrrp_type, train_type, column_group_size, cfm_input_dim, use_cfm, experiments_path=None, runs=1):
    
    datetime_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = os.path.join(experiments_path, f'model/{model_name}/{train_type}')
    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)

    history_path = os.path.join(experiments_path, 'history')
    if not os.path.exists(history_path):
        os.makedirs(history_path, exist_ok=True)

    # 检查点路径
    checkpoint_path = os.path.join(model_path, 'checkpoint.pth')
    
    # 尝试加载检查点
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint:
        start_run_idx = checkpoint['run_idx']
        all_runs_history = checkpoint['all_runs_history']
    else:
        start_run_idx = 0
        all_runs_history = {
            'runs': [],
            'best_run': None,
            'best_accuracy': 0.0,
            'best_run_index': 0,
            'mean_accuracy': 0.0,
            'std_accuracy': 0.0,
            'config': {
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
                'use_cfm': use_cfm,
                'runs': runs
            }
        }

    for run_idx in range(start_run_idx, runs):
        logging.info(f'Starting run {run_idx + 1}/{runs}')
        
        train_set = load_dataset('dataset', True, use_cfm, dataset, data_type, data_hrrp_type, batch_size, column_group_size)
        valid_set = load_dataset('dataset', False, use_cfm, dataset, data_type, data_hrrp_type, batch_size, column_group_size)
        print(f'Train set size: {len(train_set.dataset)}')
        print(f'Validation set size: {len(valid_set.dataset)}')

        m = model.Model(
            classes=classes, dropout_rate=dropout_rate, channels=channels,
            lr=lr, lr_step=lr_step, lr_decay=lr_decay,
            weight_decay=weight_decay, cfm_input_dim=cfm_input_dim, use_cfm=use_cfm
        )

        # 如果是恢复训练且是第一个要恢复的run
        if checkpoint and run_idx == start_run_idx:
            logging.info(f'Loading checkpoint from run {run_idx + 1}, epoch {checkpoint["epoch"] + 1}')
            m.net.load_state_dict(checkpoint['model_state_dict'])
            m.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if checkpoint['scheduler_state_dict'] and m.lr_scheduler:
                m.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
        else:
            start_epoch = 0

        run_history = {
            'loss': [],
            'accuracy': [],
            'best_accuracy': 0.0,
            'best_epoch': 0
        }

        best_accuracy = 0.0
        best_epoch = 0

        try:
            for epoch in range(start_epoch, epochs):
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
                    current_lr = m.lr_scheduler.get_last_lr()[0]
                    m.lr_scheduler.step()
                    accuracy = validation(m, valid_set)
                    logging.info(
                        f'Run {run_idx + 1}/{runs} | Epoch: {epoch + 1:03d}/{epochs:03d} | loss={np.mean(_loss):.4f} | lr={current_lr:.6f} | accuracy={accuracy:.2f}'
                    )
                else:
                    accuracy = validation(m, valid_set)
                    logging.info(
                        f'Run {run_idx + 1}/{runs} | Epoch: {epoch + 1:03d}/{epochs:03d} | loss={np.mean(_loss):.4f} | accuracy={accuracy:.2f}'
                    )

                run_history['loss'].append(np.mean(_loss))
                run_history['accuracy'].append(accuracy)
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_epoch = epoch + 1
                    m.save(os.path.join(model_path, f'model-run{run_idx}-best.pth'))

                # 每个epoch结束后保存检查点
                save_checkpoint(checkpoint_path, run_idx, epoch, m.net, m.optimizer, m.lr_scheduler, all_runs_history)

        except Exception as e:
            logging.error(f'Training interrupted at run {run_idx + 1}, epoch {epoch + 1}: {str(e)}')
            raise e

        run_history['best_accuracy'] = best_accuracy
        run_history['best_epoch'] = best_epoch
        
        if best_accuracy > all_runs_history['best_accuracy']:
            all_runs_history['best_accuracy'] = best_accuracy
            all_runs_history['best_run_index'] = run_idx
            all_runs_history['best_run'] = run_history
            m.save(os.path.join(model_path, f'model-overall-best.pth'))

        all_runs_history['runs'].append(run_history)

    # 计算所有运行的统计信息
    all_accuracies = [run['best_accuracy'] for run in all_runs_history['runs']]
    all_runs_history['mean_accuracy'] = np.mean(all_accuracies)
    all_runs_history['std_accuracy'] = np.std(all_accuracies)
    
    # 保存历史记录
    with open(os.path.join(history_path, f'history-{model_name}-{train_type}-{datetime_str}.json'), mode='w', encoding='utf-8') as f:
        json.dump(all_runs_history, f, ensure_ascii=True, indent=2)
    
    # 训练完成后删除检查点
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)


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
    
    # 从配置文件中读取runs参数，默认为1
    runs = config.get('runs', 1)
    
    run(epochs, dataset, classes, channels, batch_size,
        lr, lr_step, lr_decay, weight_decay, dropout_rate,
        model_name, data_type, data_hrrp_type, train_type, column_group_size, cfm_input_dim, use_cfm, experiments_path, runs)

    logging.info('Finish')


if __name__ == '__main__':
    app.run(main)
