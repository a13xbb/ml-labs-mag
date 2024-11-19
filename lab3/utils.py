import pandas as pd

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