from absl import logging
from absl import flags
from absl import app

from multiprocessing import Pool
from PIL import Image
import numpy as np

import json
import glob
import os

import mstar
from hrrp_generate import HRRPGenerator

import pdb
import matplotlib.pyplot as plt

flags.DEFINE_string('image_root', default='dataset', help='')
flags.DEFINE_string('dataset', default='soc', help='')
flags.DEFINE_string('type', default='sar', help='')
flags.DEFINE_boolean('is_train', default=False, help='')
flags.DEFINE_integer('chip_size', default=128, help='')
flags.DEFINE_integer('patch_size', default=128, help='')
flags.DEFINE_boolean('use_phase', default=False, help='')
flags.DEFINE_integer('column_group_size', default=1, help='Number of columns to average in HRRP')

FLAGS = flags.FLAGS

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def data_scaling(chip):
    r = chip.max() - chip.min()
    scaled = (chip - chip.min()) / r
    scaled = (scaled * 255).astype(np.uint8)
    return scaled[:,:,0]


def log_scale(chip):
    return np.log10(np.abs(chip) + 1)

def enhancement(img):
    pixel_count = np.zeros(256, dtype=int)
    for i in range(256):
        pixel_count[i] = np.sum(img == i)  # 统计图像中灰度i出现的次数
    
    max_count = np.argmax(pixel_count)  # 灰度i出现的次数最大值的下标，即此时的i
    min_count = np.argmin(pixel_count)  # 灰度i出现的次数最小值的下标，即此时的i
    
    if min_count > max_count:
        threshVal = min_count - max_count
        img = 255.0 / threshVal * img  # 灰度 = 原灰度 * 255/threshVal
    else:
        img = 3 * img  # 灰度 = 原灰度 * 3
    
    img[img > 255] = 255  # 大于255的灰度记为255
    img[img < 255] += 0.5
    enhanced_img = np.round(img).astype(np.uint8)
    
    return enhanced_img

def amplitude_to_grayscale(amplitude):
    
    # 展平数组以便处理
    amplitude_flat = amplitude.flatten()
    
    max_pixel = np.max(amplitude_flat)  # 图像中最大的幅度值
    min_pixel = np.min(amplitude_flat)  # 图像中最小的幅度值
    pixel_range = max_pixel - min_pixel
    pixel_scale = 255.0 / pixel_range   # 变换比例
    
    amplitude1 = amplitude_flat - min_pixel
    amplitude2 = amplitude1 * pixel_scale
    amplitude3 = np.round(amplitude2 + 0.5).astype(np.uint8)
    
    pic = amplitude3.reshape((128, 128))
    I = (pic / 255.0).astype(np.float32)  # 将像素值归一化到0-1之间
    
    enhanced_img = enhancement(pic)
    enI = (enhanced_img / 255.0).astype(np.float32)  # 将像素值归一化到0-1之间

    # # 显示原始图像
    # plt.figure(figsize=(6, 3))
    # plt.subplot(1, 2, 1)
    # plt.imshow(I, cmap='gray')
    # plt.title('before enhancement')
    # plt.axis('off')


    # # 显示增强后的图像
    # plt.subplot(1, 2, 2)
    # plt.imshow(enI, cmap='gray')
    # plt.title('after enhancement')
    # plt.axis('off')
    
    # plt.savefig('enhanced_image.png', dpi=300, bbox_inches='tight')

    return enI

def plot_1d_data(data, save_path):
    plt.figure(figsize=(8, 4))
    plt.plot(data)
    plt.grid(True)
    plt.xlabel('Range Cell')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def generate(src_path, dst_path, is_train, chip_size, patch_size, use_phase, dataset, data_type, column_group_size):
    if not os.path.exists(src_path):
        return
    if not os.path.exists(dst_path):
        os.makedirs(dst_path, exist_ok=True)
    print(f'Target Name: {os.path.basename(dst_path)}')

    _mstar = mstar.MSTAR(
        name=dataset, is_train=is_train, chip_size=chip_size, patch_size=patch_size, use_phase=use_phase, stride=1
    )
    _generator = HRRPGenerator(chip_size=128, patch_size=128, use_phase=False)
    
    image_list = glob.glob(os.path.join(src_path, '*'))

    for path in image_list:
        label, _images = _mstar.read(path)
        for i, _image in enumerate(_images):
            name = os.path.splitext(os.path.basename(path))[0]
            if data_type == 'hrrp_column':
                _generator.process_hrrp_column(_image, label, name, dst_path, column_group_size, i)
            elif data_type == 'hrrp_column_complex':
                _generator.process_hrrp_column_complex(_image, label, name, dst_path, column_group_size, i)
            else:
                # _image = log_scale(_image)
                # pdb.set_trace()
                if data_type == 'sar':
                    _image_amp = amplitude_to_grayscale(_image)
                    if not use_phase:
                        _image = np.expand_dims(_image_amp, axis=2)
                with open(os.path.join(dst_path, f'{name}-{i}.json'), mode='w', encoding='utf-8') as f:
                    json.dump(label, f, ensure_ascii=False, indent=2)
                np.save(os.path.join(dst_path, f'{name}-{i}.npy'), _image)

                Image.fromarray(data_scaling(_image)).convert('L').save(os.path.join(dst_path, f'{name}-{i}.bmp'))


def main(_):
    dataset_root = os.path.join(project_root, FLAGS.image_root, FLAGS.dataset)
    raw_root = os.path.join(dataset_root, 'raw')

    mode = 'train' if FLAGS.is_train else 'test'
    output_root = os.path.join(dataset_root, mode, FLAGS.type)
    if FLAGS.type == 'hrrp_column' or FLAGS.type == 'hrrp_row' or FLAGS.type == 'hrrp_column_complex' or FLAGS.type == 'hrrp_row_complex':
        output_root = os.path.join(output_root, str(FLAGS.column_group_size))
        
    if not os.path.exists(output_root):
        os.makedirs(output_root, exist_ok=True)

    arguments = [
        (
            os.path.join(raw_root, mode, target),
            os.path.join(output_root, target),
            FLAGS.is_train, FLAGS.chip_size, FLAGS.patch_size, FLAGS.use_phase, FLAGS.dataset, FLAGS.type,
            FLAGS.column_group_size
        ) for target in mstar.target_name[FLAGS.dataset]
    ]

    with Pool(10) as p:
        p.starmap(generate, arguments)
    # for args in arguments:
    #     generate(*args)


if __name__ == '__main__':
    app.run(main)
