# -*- coding: utf-8 -*-
# Author: Subhasis Ray <ray dot subhasis at gmail dot com>
# Created: 2020-07-07 7:37 PM
import cv2
import numpy as np
import os
from pycocotools.coco import COCO

ann_file = 'C:/Users/raysu/Documents/src/argos_data/tmp/training/annotations.json'
img_dir = os.path.dirname(ann_file)

coco = COCO(ann_file)

im_ids = list(coco.imgToAnns.keys())

for im_id in im_ids:
    ann_ids = [x for x in coco.getAnnIds(imgIds=im_id)]
    target = [x for x in coco.loadAnns(ann_ids) if x['image_id'] == im_id]
    print(f'Image id: {im_id}, {type(im_id)}, annotation ids: {ann_ids}')

    coco_imgs = coco.loadImgs(im_id)
    print('Coco images', coco_imgs)
    fname = coco_imgs[0]['file_name']
    fpath = os.path.join(img_dir, fname)
    print('Reading image', fpath)
    img = cv2.imread(fpath)
    cv2.destroyAllWindows()
    cv2.imshow('image', img)

    masks = [coco.annToMask(x) for x in target]
    for ii, mask in enumerate(masks):
        cv2.imshow(f'Mask {ii}', mask*255)

    mask = np.stack(masks).mean(axis=0) * 255
    mask = mask.astype('uint8')
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_ = img.copy()
    _ = cv2.drawContours(img_, contours, -1, (0, 0, 255))
    cv2.imshow('Contours', img_)
    key = cv2.waitKey(1000)