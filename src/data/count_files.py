import os
import glob
from collections import defaultdict
import json
from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string('dataset_root', 'dataset', '数据集根目录')
flags.DEFINE_string('dataset_name', 'soc', '数据集名称')
flags.DEFINE_string('mode', 'train', '模式：train 或 test')
flags.DEFINE_string('data_type', 'sar1', '数据类型')
flags.DEFINE_string('target_class', None, '目标类别，如果不指定则统计所有类别')

def count_files(root_path, class_name=None):
    """统计指定目录下的文件数量"""
    stats = defaultdict(int)
    class_stats = defaultdict(lambda: defaultdict(int))
    
    # 如果指定了类别，只统计该类别
    if class_name:
        path = os.path.join(root_path, class_name)
        if os.path.exists(path):
            files = glob.glob(os.path.join(path, '*'))
            for f in files:
                ext = os.path.splitext(f)[1]
                stats[ext] += 1
                # 获取文件的基本名称（不包含扩展名和索引）
                base_name = os.path.basename(f).split('-')[0]
                class_stats[class_name][base_name] += 1
    else:
        # 获取所有类别目录
        class_dirs = [d for d in os.listdir(root_path) 
                     if os.path.isdir(os.path.join(root_path, d))]
        
        for class_dir in class_dirs:
            path = os.path.join(root_path, class_dir)
            files = glob.glob(os.path.join(path, '*'))
            print(f"\n类别 {class_dir}:")
            class_total = defaultdict(int)
            
            for f in files:
                ext = os.path.splitext(f)[1]
                stats[ext] += 1
                class_total[ext] += 1
                # 获取文件的基本名称（不包含扩展名和索引）
                base_name = os.path.basename(f).split('-')[0]
                class_stats[class_dir][base_name] += 1
            
            # 打印每个类别的统计信息
            print(f"  - 总文件数: {len(files)}")
            for ext, count in class_total.items():
                print(f"  - {ext} 文件数: {count}")
            print(f"  - 不同样本数: {len(class_stats[class_dir])}")
    
    return stats, class_stats

def main(_):
    # 构建数据集路径
    dataset_path = os.path.join(
        FLAGS.dataset_root,
        FLAGS.dataset_name,
        FLAGS.mode,
        FLAGS.data_type
    )
    
    print(f"\n统计路径: {dataset_path}")
    
    # 统计文件
    stats, class_stats = count_files(dataset_path, FLAGS.target_class)
    
    if FLAGS.target_class:
        print(f"\n{FLAGS.target_class} 类别统计:")
        print(f"总文件数: {sum(stats.values())}")
        for ext, count in stats.items():
            print(f"{ext} 文件数: {count}")
        print(f"不同样本数: {len(class_stats[FLAGS.target_class])}")
    else:
        print("\n总体统计:")
        print(f"总文件数: {sum(stats.values())}")
        for ext, count in stats.items():
            print(f"{ext} 文件数: {count}")
        
        print("\n每个类别的样本数:")
        for class_name, samples in class_stats.items():
            print(f"{class_name}: {len(samples)} 个不同样本")

if __name__ == '__main__':
    app.run(main) 