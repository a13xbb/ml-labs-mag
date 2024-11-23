import pandas as pd
import torch
import torch.nn.functional as F
import albumentations as A
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from random import randint
from torchvision.transforms import v2
import torchvision.transforms as transforms
import math

def create_dataset(images_df: pd.DataFrame, annotations_df: pd.DataFrame) -> dict:
    train_dataset = []
    for _, img_row in images_df.iterrows():
        image_id = img_row['image_id']
        file_name = img_row['file_name']
        
        img_annotations = annotations_df[annotations_df['image_id'] == image_id]
        
        annotations_list = []
        for _, ann_row in img_annotations.iterrows():
            annotations_list.append({
                'class' : ann_row['category_id'],
                'bbox' : ann_row['bbox'],
                'area' : ann_row['area']
            })
        
        train_dataset.append({
            'file_name' : file_name,
            'annotations' : annotations_list
        })
    return train_dataset


class LocalContrastNorm:
    """Local Contrast Normalization"""
    def __init__(self, kernel_size=3, eps=1e-5):
        self.kernel_size = kernel_size
        self.eps = eps
        
    def __call__(self, pic):
        mean = F.avg_pool2d(pic, self.kernel_size, stride=1, padding=self.kernel_size//2)
        std = torch.sqrt(F.avg_pool2d((pic - mean)**2, self.kernel_size, stride=1, padding=self.kernel_size//2) + self.eps)
        std[std < 1] = 1
        return (pic - mean) / std
    

def plot_image(image, bboxes, labels, classes_colors):
    for i, bbox in enumerate(bboxes):
        xmin, ymin, w, h = bbox
        draw = ImageDraw.Draw(image)
        draw.rectangle([xmin, ymin, xmin + w, ymin + h], outline=classes_colors[labels[i]], width=3)
    plt.imshow(image)
    plt.show()


def rotate_point(point: tuple, old_img_size: tuple, new_img_size: tuple,  angle: float):
    x, y = point
    w, h = old_img_size
    angle_rad = math.radians(-angle)
    
    cx, cy = w / 2, h / 2
    
    new_width, new_height = new_img_size
    # new_width = int(abs(w * math.cos(angle_rad)) + abs(h * math.sin(angle_rad)))
    # new_height = int(abs(w * math.sin(angle_rad)) + abs(h * math.cos(angle_rad)))
    
    new_cx = new_width / 2
    new_cy = new_height / 2
    
    translated_x = x - cx
    translated_y = y - cy
    
    rotated_x = translated_x * math.cos(angle_rad) - translated_y * math.sin(angle_rad)
    rotated_y = translated_x * math.sin(angle_rad) + translated_y * math.cos(angle_rad)

    final_x = rotated_x + new_cx
    final_y = rotated_y + new_cy
    
    return final_x, final_y


def rotate_bbox(bbox, old_img_size: tuple, new_img_size: tuple, angle: float):
    x_min, y_min, w, h = bbox
    points = [
        (x_min, y_min),
        (x_min, y_min + h),
        (x_min + w, y_min + h),
        (x_min + w, y_min)
    ]
    
    rotated_x = []
    rotated_y = []
    for point in points:
        r_x, r_y = rotate_point(point, old_img_size, new_img_size, angle)
        rotated_x.append(r_x)
        rotated_y.append(r_y)
    
    x_min = min(rotated_x)
    y_min = min(rotated_y)
    w = max(rotated_x) - x_min
    h = max(rotated_y) - y_min
    
    return [x_min, y_min, w, h]


# def BasicTransform(img, bboxes):
#     # img  = v2.RandomHorizontalFlip()(img)
#     # for i, bbox in enumerate(bboxes):
#     #     bboxes[i] = torch.tensor(rotate_bbox(bbox, img.size, img.size, 180),
#     #                              dtype=torch.float32)
    
#     angle = randint(-30, 30)
#     r_image = img.rotate(angle, expand=1)
#     for i, bbox in enumerate(bboxes):
#         bboxes[i] = torch.tensor(rotate_bbox(bbox, img.size, r_image.size, angle),
#                                  dtype=torch.float32)
    
#     r_image = v2.ColorJitter(brightness=(0.4, 2), contrast=(1, 6),
#             saturation=(0, 2.5), hue=(-0.25,0.25))(r_image)
    
#     r_image = v2.RandomInvert(0.3)(r_image)
    
#     resized_image = transforms.Resize((224, 224))(r_image)
#     w_new, h_new = resized_image.size
#     w_old, h_old = r_image.size
#     scale_x = w_new / w_old
#     scale_y = h_new / h_old
#     for i, bbox in enumerate(bboxes):
#         x_old, y_old, w_old, h_old = bbox
#         bboxes[i] = torch.tensor([x_old * scale_x, y_old * scale_y, w_old * scale_x, h_old * scale_y],
#                                     dtype=torch.float32)
    
#     r_image = transforms.ToTensor()
    
#     return r_image, bboxes
        