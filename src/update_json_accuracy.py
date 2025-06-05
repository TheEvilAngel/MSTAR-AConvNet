import json
import os
import glob
from typing import List

def process_json_file(file_path):
    # 读取JSON文件
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # # 如果已经有best_accuracy，跳过处理
    # if 'best_accuracy' in data:
    #     print(f"文件 {file_path} 已经包含best_accuracy，跳过处理")
    #     return
    
    # 确保accuracy键存在
    if 'accuracy' not in data:
        print(f"文件 {file_path} 中没有找到accuracy数据")
        return
    
    # 计算最佳准确率
    best_accuracy = max(data['accuracy'])
    
    # 计算top10平均准确率
    sorted_accuracies = sorted(data['accuracy'], reverse=True)
    top10_avg_accuracy = sum(sorted_accuracies[:10]) / min(10, len(sorted_accuracies))
    
    # 添加统计值到数据中
    data['best_accuracy'] = best_accuracy
    data['top10_avg_acc'] = top10_avg_accuracy
    
    # 保存更新后的JSON文件
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"文件 {file_path}:")
    print(f"  - best_accuracy: {best_accuracy:.4f}")
    print(f"  - top10_avg_acc: {top10_avg_accuracy:.4f}")

def main():
    # 获取所有JSON文件
    json_files = glob.glob('experiments/history/*.json')
    
    # 处理每个文件
    for file_path in json_files:
        try:
            process_json_file(file_path)
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {str(e)}")

if __name__ == "__main__":
    main() 