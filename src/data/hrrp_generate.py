from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import mstar
import pdb

class HRRPGenerator:
    def __init__(self, chip_size=128, patch_size=128, use_phase=False):
        """
        初始化HRRP生成器
        
        参数:
            chip_size: 图像芯片大小
            patch_size: 图像块大小
            use_phase: 是否使用相位信息
        """
        self.chip_size = chip_size
        self.patch_size = patch_size
        self.use_phase = use_phase
        
    @staticmethod
    def data_scaling(chip):
        """数据缩放到0-255范围"""
        r = chip.max() - chip.min()
        scaled = (chip - chip.min()) / r
        scaled = (scaled * 255).astype(np.uint8)
        return scaled
    
    @staticmethod
    def plot_1d_data(data, save_path):
        """绘制一维HRRP数据"""
        plt.figure(figsize=(8, 4))
        plt.plot(data)
        plt.grid(True)
        plt.xlabel('Range Cell')
        plt.ylabel('Amplitude')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
    def process_hrrp_column(self, image, label, name, dst_path, column_group_size, index):
        """处理列方向HRRP"""
        # FFT变换
        _image = np.fft.ifft(image, axis=1)  # column is cross-range
        _image = np.abs(_image)
        _image = _image / np.max(_image, axis=0, keepdims=True)  # normalize column
        
        # 保存BMP图像
        Image.fromarray(self.data_scaling(_image[:,:,0])).convert('L').save(
            os.path.join(dst_path, f'{name}-{index}.bmp')
        )
        
        # 分组并平均
        num_columns = _image.shape[1]
        num_groups = num_columns // column_group_size
        
        for j in range(num_groups):
            start_col = j * column_group_size
            end_col = (j + 1) * column_group_size
            group_avg = np.mean(_image[:, start_col:end_col], axis=1)
            
            # 保存标签
            with open(os.path.join(dst_path, f'{name}-{index}-{j}.json'), mode='w', encoding='utf-8') as f:
                json.dump(label, f, ensure_ascii=False, indent=2)
            # 保存一维数据
            np.save(os.path.join(dst_path, f'{name}-{index}-{j}.npy'), group_avg)
            
            # 绘制并保存图像
            self.plot_1d_data(group_avg, os.path.join(dst_path, f'{name}-{index}-{j}.png'))
    
    @staticmethod
    def taylor_window(N, nbar=4, sll=-30):
        """生成Taylor窗
        
        参数:
            N: 窗长度
            nbar: 过渡区域的近似切比雪夫函数的数量
            sll: 目标旁瓣电平(dB)
        """
        def calculate_fm(m, sp2):
            n = np.arange(1, nbar)
            return np.prod(1 - (m**2)/(sp2[n-1]))
            
        def calculate_wn(n, sigma, nbar):
            return ((-1)**(n+1) * 
                   np.prod(1-n**2/np.array([m**2 for m in range(1, nbar)])) /
                   (2 * np.prod(np.array([1-n**2/m**2 for m in range(nbar, 2*nbar-1)]))))

        A = np.arccosh(10**(-sll/20))
        sp2 = (nbar**2 * (np.cosh(A/nbar)**2))
        
        # 计算权重
        w = [1]
        for n in range(1, nbar):
            w.append(calculate_wn(n, sp2, nbar))
        
        # 生成窗
        x = np.arange(N) - (N-1)/2
        window = w[0]
        for n in range(1, nbar):
            window += 2*w[n]*np.cos(2*np.pi*n*x/N)
        
        return window/np.max(window)

    def process_hrrp_column_complex(self, image, label, name, dst_path, column_group_size, index):
        """处理复数SAR图像并生成列方向HRRP
        
        参数:
            image: [H,W,2]数组，其中[:,:,0]是幅度，[:,:,1]是相位
            label: 标签信息
            name: 文件名
            dst_path: 保存路径
            column_group_size: 列分组大小
            index: 索引
        """
        Image.fromarray(self.data_scaling(image[:,:,0])).convert('L').save(os.path.join(dst_path, f'{name}-{index}-origin.bmp'))

        # 1. 将幅度和相位转换为复数
        amplitude = image[:,:,0]
        phase = image[:,:,1]
        complex_image = amplitude * np.exp(1j * phase)
        
        # 2. 进行2D IFFT
        ifft_image = np.fft.ifft2(complex_image)
        ifft_image = np.fft.fftshift(ifft_image)  # 将零频率分量移到中心
        
        # 3. 去除Taylor窗并进行zero padding
        H, W = ifft_image.shape
        
        # 创建2D Taylor窗
        taylor_win_h = self.taylor_window(H, nbar=4, sll=-30)
        taylor_win_w = self.taylor_window(W, nbar=4, sll=-30)
        taylor_window_2d = np.outer(taylor_win_h, taylor_win_w)
        
        # 去除窗函数影响
        unwindowed = ifft_image / (taylor_window_2d + 1e-10)  # 添加小量避免除零
        
        # Zero padding - 保持原始大小的2倍
        # # TODO: zero padding 有什么用
        # pad_h = H
        # pad_w = W
        # padded = np.pad(unwindowed, 
        #                ((pad_h//2, pad_h//2), (pad_w//2, pad_w//2)), 
        #                mode='constant')
        padded = unwindowed
        
        # 4. 进行2D FFT并保存图像
        fft_image = np.fft.fft2(padded)
        fft_magnitude = np.abs(fft_image)
        
        # 归一化到0-255
        fft_normalized = ((fft_magnitude - fft_magnitude.min()) / 
                        (fft_magnitude.max() - fft_magnitude.min()) * 255).astype(np.uint8)
        
        # 保存FFT结果为BMP
        Image.fromarray(fft_normalized).convert('L').save(
            os.path.join(dst_path, f'{name}-{index}-fft2d.bmp')
        )
        # 5. 进行HRRP列方向处理
        # 使用原始的process_hrrp_column方法处理
        # FFT变换
        _image = np.fft.ifft(fft_image, axis=1)  # column is cross-range
        _image = np.abs(_image)
        _image = _image / np.max(_image, axis=0, keepdims=True)  # normalize column
        
        # 保存BMP图像
        Image.fromarray(self.data_scaling(_image)).convert('L').save(
            os.path.join(dst_path, f'{name}-{index}-hrrp2d.bmp')
        )
        
        # 分组并平均
        num_columns = _image.shape[1]
        num_groups = num_columns // column_group_size
        
        for j in range(num_groups):
            start_col = j * column_group_size
            end_col = (j + 1) * column_group_size
            group_avg = np.mean(_image[:, start_col:end_col], axis=1)
            
            # 保存标签
            with open(os.path.join(dst_path, f'{name}-{index}-{j}.json'), mode='w', encoding='utf-8') as f:
                json.dump(label, f, ensure_ascii=False, indent=2)
            
            # 保存一维数据
            np.save(os.path.join(dst_path, f'{name}-{index}-{j}.npy'), group_avg)
            
            # 绘制并保存图像
            self.plot_1d_data(group_avg, os.path.join(dst_path, f'{name}-{index}-{j}.png'))

    
    def process_hrrp_row(self, image, label, name, dst_path, column_group_size, index):
        """处理行方向HRRP"""
        # FFT变换
        _image = np.fft.ifft(image, axis=0)  # row is cross-range
        _image = np.abs(_image)
        _image = _image / np.max(_image, axis=1, keepdims=True)  # normalize row
        
        # 保存BMP图像
        Image.fromarray(self.data_scaling(_image[:,:,0])).convert('L').save(
            os.path.join(dst_path, f'{name}-{index}.bmp')
        )
        
        # 分组并平均
        num_rows = _image.shape[0]
        num_groups = num_rows // column_group_size
        
        for j in range(num_groups):
            start_row = j * column_group_size
            end_row = (j + 1) * column_group_size
            group_avg = np.mean(_image[start_row:end_row, :], axis=0)
            
            # 保存标签
            with open(os.path.join(dst_path, f'{name}-{index}-{j}.json'), mode='w', encoding='utf-8') as f:
                json.dump(label, f, ensure_ascii=False, indent=2)
            
            # 保存一维数据
            np.save(os.path.join(dst_path, f'{name}-{index}-{j}.npy'), group_avg)
            
            # 绘制并保存图像
            self.plot_1d_data(group_avg, os.path.join(dst_path, f'{name}-{index}-{j}.png'))