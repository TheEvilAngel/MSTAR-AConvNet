import json
import os
import glob
import csv

def get_config_value(config, key):
    """从配置中获取值，如果不存在则返回'-'"""
    return str(config.get(key, '-'))

def format_accuracy(data, key):
    """格式化准确率数值，如果不存在或无效则返回'-'"""
    try:
        value = data.get(key)
        if value is not None:
            return f"{float(value):.2f}"
    except (ValueError, TypeError):
        pass
    return '-'

def generate_results_table():
    # 获取所有JSON文件
    json_files = glob.glob('experiments/history/*.json')
    
    # 定义CSV的表头
    headers = [
        'dataset', 'data_type', 'train_type', 'use_cfm', 'data_hrrp_type',
        'column_group_size', 'cfm_type', 'best_accuracy',
        'mean_accuracy', 'std_accuracy'
    ]
    
    # 创建results目录（如果不存在）
    os.makedirs('experiments/results', exist_ok=True)
    
    # 创建并写入CSV文件
    with open('experiments/results/experiment_results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # 写入表头
        writer.writerow(headers)
        
        # 处理每个JSON文件
        for json_file in json_files:
            print(f"正在处理文件: {json_file}")
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"JSON解析错误 {json_file}: {str(e)}")
                continue
            except Exception as e:
                print(f"读取文件错误 {json_file}: {str(e)}")
                continue
            
            # 获取配置参数
            config = data.get('config', {})  # 如果没有config字段，使用空字典
            
            # 准备一行数据
            row = [
                get_config_value(config, 'dataset'),
                get_config_value(config, 'data_type'),
                get_config_value(config, 'train_type'),
                get_config_value(config, 'use_cfm'),
                get_config_value(config, 'data_hrrp_type'),
                get_config_value(config, 'column_group_size'),
                get_config_value(config, 'cfm_type'),
                format_accuracy(data, 'best_accuracy'),
                format_accuracy(data, 'mean_accuracy'),
                format_accuracy(data, 'std_accuracy')
            ]
            
            # 写入一行数据
            writer.writerow(row)
            print(f"成功处理文件!")

if __name__ == "__main__":
    generate_results_table() 