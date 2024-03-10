import json
import shutil
import os

if __name__ == '__main__':
    train_dir = 'data/custom/train2017'
    val_dir = 'data/custom/val2017'
    
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
        
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
        
    with open('annotations/custom_train.json') as file:
        training = json.load(file)
        
    training_set = set()
    for image in training['images']:
        training_set.add(image['file_name'])
    
    with open('annotations/custom_val.json') as file:
        validation = json.load(file)
        
    validation_set = set()
    for image in validation['images']:
        validation_set.add(image['file_name'])
        
    for file_name in training_set:
        source_file = f"data/images/{file_name}"
        shutil.copy2(source_file, f"{train_dir}/{file_name}")
        
    for file_name in validation_set:
        source_file = f"images/{file_name}"
        shutil.copy2(source_file, f"{val_dir}/{file_name}")