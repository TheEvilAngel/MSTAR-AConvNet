import json
import os
import glob
import matplotlib.pyplot as plt

def plot_and_save_curves(json_file):
    # 读取JSON文件
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # 获取文件名(不包含扩展名和时间戳)
    base_name = os.path.splitext(os.path.basename(json_file))[0]
    # 移除时间戳部分（假设时间戳格式为-YYYYMMDD_HHMMSS）
    base_name = base_name.rsplit('-', 1)[0]  # 从右边分割，只分割一次
    
    # 创建保存图片的目录
    save_dir = 'experiments/history/pic'
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取best_run数据
    best_run = data['best_run']
    epochs = list(range(1, len(best_run['accuracy']) + 1))
    
    # 创建图形和双y轴
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # 绘制accuracy曲线（使用左y轴）
    line1 = ax1.plot(epochs, best_run['accuracy'], 'b-', label='Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    # 绘制loss曲线（使用右y轴）
    ax2 = ax1.twinx()
    line2 = ax2.plot(epochs, best_run['loss'], 'r-', label='Loss')
    ax2.set_ylabel('Loss', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    # 合并两条线的图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')
    
    # 创建包含统计信息的标题
    title = f'{base_name}\n'
    title += f'Best Acc: {data["best_accuracy"]:.2f}% | '
    title += f'Mean Acc: {data["mean_accuracy"]:.2f}% | '
    title += f'Std Acc: {data["std_accuracy"]:.2f}%'
    
    plt.title(title)
    plt.grid(True)
    
    # 保存图片
    plt.savefig(os.path.join(save_dir, f'{base_name}_metrics.png'), bbox_inches='tight', dpi=300)
    plt.close()

def main():
    # 获取所有JSON文件
    json_files = glob.glob('experiments/history/*.json')
    
    # 处理每个文件
    for json_file in json_files:
        try:
            print(f"正在处理文件: {json_file}")
            plot_and_save_curves(json_file)
            print(f"成功生成图片!")
        except Exception as e:
            print(f"处理文件 {json_file} 时出错: {str(e)}")

if __name__ == "__main__":
    main() 