from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
import cv2
import numpy as np
import pandas as pd
import os
import glob

img_path ='/root/home/mmdetection/data/pseudo_lung/train'

png_list = glob.glob(os.path.join(img_path, '*.png'))
jpg_list = glob.glob(os.path.join(img_path, '*.jpg'))
train_list = png_list + jpg_list

print(len(train_list))


config_file = '/root/home/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco_sample.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
# url: https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
checkpoint_file = '/root/home/mmdetection/work_dirs/faster_rcnn_r50_fpn_1x_coco_sample/latest.pth'

model = init_detector(config_file, checkpoint_file, device='cuda:0')




annotations = []
images = []
obj_count = 0

for idx, path in enumerate(mmcv.track_iter_progress(train_list)):
    
    filename = path.split('/')[-1]
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    height, width = image.shape[:2]

    images.append(dict(
            id=idx,
            file_name=filename,
            height=height,
            width=width))

    #####   annotation
    anno_list =[]

    result = inference_detector(model, path)

    for lung in result[0]:
        ## 그 confidence가 0.85를 넘을때~~
        if lung[-1] >= 0.85:
            
            bounding_box = lung[:-1]
            anno_list.append(bounding_box)
        else:
            
            print('\nconfidence is too low, under 0.85')


    bboxes = []
    labels = []
    masks = []

    for (x, y, w, h) in anno_list:
        # print(x, y, w, h)
        data_anno = dict(
                image_id=idx,
                id=obj_count,
                category_id=0,
                bbox=[x, y, w, h],
                area=(w * h),
                iscrowd=0)
        annotations.append(data_anno)
        obj_count += 1

coco_format_json = dict(
    images=images,
    annotations=annotations,
    categories=[{'id':0, 'name': 'lung'}])
mmcv.dump(coco_format_json, '/root/home/mmdetection/data/pseudo_lung/train/annotation_coco.json')

