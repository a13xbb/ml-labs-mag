import pandas as pd
import torch
import torch.nn.functional as F
import albumentations as A
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

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