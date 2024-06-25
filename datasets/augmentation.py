import os
from typing import Any

import cv2
import random
import numpy as np
from PIL import Image, ImageFilter

class RandomCropEdge(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            """应用随机裁剪"""
            h,w,_ = img.shape
            if random.random() < 0.5:
                crop_start_x = np.random.randint(0, int(0.1*w)+1)
                crop_start_y = np.random.randint(0, int(0.1*h)+1)
                masked_image = img[crop_start_x: crop_start_x + int(0.9 * w) - 1, crop_start_y: crop_start_y + int(0.9 * h) - 1]
            else:
                crop_start_x = np.random.randint(int(0.9*w)-1, w)
                crop_start_y = np.random.randint(int(0.9*h)-1, h)
                masked_image = img[crop_start_x - int(0.9 * w) + 1: crop_start_x, crop_start_y - int(0.9 * h) + 1: crop_start_y]

            # 创建遮挡区域并应用于图像
            img = cv2.resize(masked_image,(80,80))

        return img


class RandomLight(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            """应用光照叠加"""
            # 随机生成alpha和beta值
            alpha = np.random.uniform(0.5, 1.5)
            beta = np.random.uniform(-0.5, 0.5)

            # 应用亮度和对比度增强
            augmented_image = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
            augmented_image = cv2.convertScaleAbs(augmented_image, alpha=alpha)

            # 将原始图像和增强后的图像进行混合
            img = cv2.addWeighted(img, 1 - 0.5, augmented_image, 0.5, 0)

        return img


class RandomBlur(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            p1 = random.random()
            """应用模糊增强"""
            if p1 < 0.5:
                kernel = np.ones((5, 5), np.float32) / (5 ** 2)

                # 应用卷积操作
                img = cv2.filter2D(img, -1, kernel)
            else:
                kernel = np.ones((3, 3), np.float32) / (3 ** 2)

                # 应用卷积操作
                img = cv2.filter2D(img, -1, kernel)

        return img


class RandomMask(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            """应用随机遮挡"""
            # 随机生成遮挡区域的位置和大小
            height, width = img.shape[:2]
            mask_width = np.random.randint(1, 64)
            mask_height = np.random.randint(1, 64)
            x1 = np.random.randint(0, width - mask_width)
            y1 = np.random.randint(0, height - mask_height)
            x2 = x1 + mask_width
            y2 = y1 + mask_height

            # 创建遮挡区域并应用于图像
            masked_image = img.copy()
            masked_image[y1:y2, x1:x2] = (0, 0, 0)
            img = masked_image

        return img


class RandomMask_keypoint(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            """应用随机遮挡"""
            p1 = random.random()
            if p1 < 0.5:
                # 随机生成遮挡区域的大小
                mask_height = int(np.random.randint(1, 30) / 2)
                x1 = 0
                y1 = 90 - mask_height
                x2 = 128
                y2 = 128

                # 创建遮挡区域并应用于图像
                masked_image = img.copy()
                masked_image[y1:y2, x1:x2] = (0, 0, 0)
                img = masked_image

        return img


class RandomMask_edge(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            """应用随机遮挡"""
            if self.p < 0.5:
                # 随机生成遮挡区域的大小
                mask_width = np.random.randint(10, 15)
                mask_height = np.random.randint(5, 10)

                # 创建遮挡区域并应用于图像
                masked_image = img.copy()
                masked_image[0:mask_height, 0:128] = (0, 0, 0)
                masked_image[128 - mask_height:128, 0:128] = (0, 0, 0)
                masked_image[0:128, 0:mask_width] = (0, 0, 0)
                masked_image[0:128, 128 - mask_width:128] = (0, 0, 0)
                img = masked_image

        return img


class RandomRotation(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            """应用随机旋转"""
            # 随机选择旋转角度
            angle = np.random.uniform(-90, 90)

            # 获取图像尺寸
            height, width = img.shape[:2]
            center_x, center_y = width / 2, height / 2

            # 计算旋转矩阵并应用于图像
            rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
            img = cv2.warpAffine(img, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT_101)

        return img


class GaussianNoise(object):
    '''
    添加高斯噪声
     - stdDev 表示标准差用于控制强度
     - mean 一般置 0
    '''

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        n_img = img.copy()
        if random.random() < self.p:
            h, w, c = img.shape
            noise = np.random.normal(0, 25, (h, w, c))
            n_img = np.clip(img + noise, 0, 255).astype(np.uint8)
        return n_img


class SaltandPepperNoise(object):
    ''' -pro 表示噪声的密集程度'''

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        n_img = img.copy()
        if random.random() < self.p:
            SNR = random.uniform(0.5, 1.0)
            n_img = np.array(n_img)
            n_img = n_img.transpose(2, 1, 0)
            c, h, w = n_img.shape
            mask = np.random.choice((0, 1, 2), size=(1, h, w), p=[SNR, (1 - SNR) / 2., (1 - SNR) / 2.])
            mask = np.repeat(mask, c, axis=0)
            n_img[mask == 1] = 255
            n_img[mask == 2] = 0
            n_img = n_img.transpose(2, 1, 0)
            n_img = n_img if n_img.shape[-1] == 3 else cv2.cvtColor(n_img, cv2.COLOR_BGR2RGB)
        return n_img


class HorizontalStripeNoise(object):
    '''水平条纹噪音'''

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        mask_img = img.copy()
        if random.random() < self.p:
            h, _, _ = mask_img.shape
            mask = np.random.rand(h) < 0.02
            for i in range(h):
                if mask[i]:
                    stripe_intensity = np.random.randint(0, 255, (1, 1, 3))
                    mask_img[i, :, :] = stripe_intensity
        return mask_img


class VerticalStreakNoise(object):
    '''垂直条纹噪音'''

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        mask_img = img.copy()
        if random.random() < self.p:
            _, w, _ = mask_img.shape
            mask = np.random.rand(w) < 0.02
            for i in range(w):
                if mask[i]:
                    stripe_intensity = np.random.randint(0, 255, (1, 1, 3))
                    mask_img[:, i, :] = stripe_intensity
        return mask_img


class HistogramEqualisation(object):
    '''直方均衡'''

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        h_img = img.copy()
        if random.random() < self.p:
            _, _, c = img.shape
            for i in range(c):
                h_img[:, :, i] = cv2.equalizeHist(h_img[:, :, i])
        return h_img


class EdgeEhance(object):
    """
    边缘锐化
     - 对于模糊图像效果显著，如果图像清晰则 p < 0.5，反之 p>0.5
     - kernel 用 PIL.ImageFilter 预设的核，效果好
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return cv2.cvtColor(
                np.array(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).filter(ImageFilter.EDGE_ENHANCE)),
                cv2.COLOR_RGB2BGR)
        else:
            return img


class GridMask(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img) -> Any:
        w, h, _ = img.shape

        square_size_w = int(w / 5)
        square_size_h = int(h / 5)

        x = np.random.randint(0, square_size_w)
        y = np.random.randint(0, square_size_w)

        mask_image = img.copy()
        if random.random() < self.p:
            original_x = x
            for _ in range(5):
                for _ in range(5):
                    mask_image[x:x + square_size_w, y:y + square_size_h, :] = 0
                    x += square_size_w * 2
                y += square_size_h * 2
                x = original_x
        return mask_image


class RandomHSV(object):
    def __init__(self, p=0.5, hgain=0.5, sgain=0.5, vgain=0.5):
        self.p = p
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain

    def __call__(self, img):
        if random.random() < self.p:
            r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain] + 1  # random gains
            hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
            dtype = img.dtype  # uint8

            x = np.arange(0, 256, dtype=np.int16)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

            img_hsv = cv2.merge(
                (cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
            img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)  # no return needed

        return img

class RandomGray(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            # 转换为灰度图像
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 复制灰度图像三次，合并成三通道图像
            img = cv2.merge([gray_image, gray_image, gray_image])

        return img



if __name__ == '__main__':
    img_dir = r'../imgs'
    imgs_name = os.listdir(img_dir)
    trans = [

        RandomLight(p=0.5),
        RandomBlur(p=0.5),
        RandomMask(p=0.5),
        RandomRotation(p=0.5),
        GaussianNoise(p=1),
        SaltandPepperNoise(p=1),
        HorizontalStripeNoise(p=1),
        VerticalStreakNoise(p=1),
        HistogramEqualisation(p=1),
        EdgeEhance(p=1),
        GridMask(p=1),
        RandomHSV(p=1),
        RandomMask_edge(p=1),
        RandomMask_keypoint(p=1)
    ]
    for name in imgs_name:
        img_path = os.path.join(img_dir, name)
        img = cv2.imread(img_path)
        # keyimg = trans[0](img)
        # blended_image = trans[0](img)
        blurred_image = trans[1](img)
        # masked_image = trans[2](img)
        # rotated_image = trans[3](img)
        # gaussianoise_image = trans[4](img)
        # saltpeppernoise_image = trans[5](img)
        # horizontalstripenoise_image = trans[6](img)
        # verticalstreaknoise_image = trans[7](img)
        # histogramequalisation_image = trans[8](img)
        edgeehance_image = trans[9](img)
        # gridmask_image = trans[10](img)
        # hsv_img = trans[11](img)
        random_edge = trans[12](img)
        random_keypoint = trans[13](img)
        # cv2.imshow('origin_image',img)
        # cv2.imshow('blended_image',blended_image)
        # cv2.imshow('blurred_image', blurred_image)
        # cv2.imshow('masked_image', masked_image)
        # cv2.imshow('rotated_image', rotated_image)
        # cv2.imshow('gaussianoise_image', gaussianoise_image)
        # cv2.imshow('saltpeppernoise_image', saltpeppernoise_image)
        # cv2.imshow('horizontalstripenoise_image', horizontalstripenoise_image)
        # cv2.imshow('verticalstreaknoise_image', verticalstreaknoise_image)
        # cv2.imshow('histogramequalisation_image', histogramequalisation_image)
        # cv2.imshow('edgeehance_image', edgeehance_image)
        # cv2.imshow('gridmask_image', gridmask_image)
        cv2.imshow('keyimg', random_keypoint)
        cv2.waitKey(0)
        cv2.destroyAllWindows()





