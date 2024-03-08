#%%
import json
from sklearn.model_selection import train_test_split
import numpy as np
import os
import shutil

#%%
########### combine annotation
dnj = json.load(open(
    '../data/dsc180b_640x640/annotations_640x640/annotations_640x640.json'
    ))
sun = json.load(open(
    '../annotations/sunny_annotations_604.json'
    ))
kev = json.load(open(
    '../annotations/kevin_annotations.json'
    ))

out ={
    'licenses': dnj['licenses'],
    'info': dnj['info'], 
    'categories': dnj['categories'], 
    'images': [], 
    'annotations': []
}

img_idx = 1
ann_idx = 1
for _,f in enumerate([dnj,sun,kev]):
    imgs = f['images']
    anns = f['annotations']

    old_to_new = {}
    for img in imgs:
        if _ == 0 and ('Sunny' in img['file_name'] or 'kevin'in img['file_name']):
            continue
        old_to_new[img['id']] = img_idx
        img['id'] = img_idx
        out['images'] += [img]
        img_idx += 1
    for ann in anns:
        if ann['image_id'] not in old_to_new:
            continue
        ann['image_id'] = old_to_new[ann['image_id']]
        ann['id'] = ann_idx
        ann_idx += 1
        out['annotations'] += [ann]
ann_save = open('../annotations/combined_annotations_640.json','w')
json.dump(out, ann_save)
ann_save.close()  
#%%
#load the annotations for the whole dataset
f = json.load(open(
    '../annotations/combined_annotations_640.json'
    ))
print(f.keys())
# %%
train_json ={
    'licenses': f['licenses'],
    'info': f['info'], 
    'categories': f['categories'], 
    'images': [], 
    'annotations': []
}
val_json ={
    'licenses': f['licenses'],
    'info': f['info'], 
    'categories': f['categories'], 
    'images': [], 
    'annotations': []
}

# %%
train, val = train_test_split(f['images'],test_size=0.33, random_state=42)

train_ids = {}
val_ids = {}
train_img = []
val_img = []
for i,img in enumerate(train):
    train_ids[img['id']] = i
    img['id'] = i
    train_img += [img]

for i,img in enumerate(val):
    val_ids[img['id']] = i
    img['id'] = i
    val_img += [img]

t_cnt = 0
v_cnt = 0
train_ann = []
val_ann = []
for ann in f['annotations']:
    if ann['image_id'] in train_ids.keys():
        ann['image_id'] = train_ids[ann['image_id']]
        ann['id'] = t_cnt
        t_cnt += 1
        train_ann += [ann]
    elif ann['image_id'] in val_ids.keys():
        ann['image_id'] = val_ids[ann['image_id']]
        ann['id'] = v_cnt
        v_cnt += 1
        val_ann += [ann]
    else:
        print(ann)
        print('no fucking way')
# %%
#path of dir where the images are stored
og_path = '../images/'
#path of dir where the training images are going to be stored
train_path = '/Users/sunwoo/Desktop/data/custom/train2017/'
#path of dir where the validation images are going to be stored
val_path = '/Users/sunwoo/Desktop/data/custom/val2017/'
for img in train_img:
    shutil.move(og_path+img['file_name'], train_path+img['file_name'])
for img in val_img:
    shutil.move(og_path+img['file_name'], val_path+img['file_name'])
# %%
train_json['images'] = train_img
val_json['images'] = val_img
train_json['annotations'] = train_ann
val_json['annotations'] = val_ann

train_save = open('../annotations/custom_train.json','w')
json.dump(train_json, train_save)
train_save.close()

val_save = open('../annotations/custom_val.json','w')
json.dump(val_json, val_save)
val_save.close()
# %%
