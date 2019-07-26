
import os
from os.path import join
import re
import cv2
import datetime
from itertools import groupby
from skimage import measure
import numpy as np
from pycocotools import mask
from pycocotools.coco import COCO
import datetime


convert = lambda text: int(text) if text.isdigit() else text.lower()
natrual_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]

def resize_binary_mask(np_mask, new_size=(256,256)):
    np_mask = np_mask.astype(np.uint8) * 255
    np_mask = cv2.resize(np_mask, new_size)
    return np_mask.astype(np.bool)

def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour

def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
                counts.append(0)
        counts.append(len(list(elements)))

    return rle

def binary_mask_to_polygon(binary_mask, tolerance=0):
    """Converts a binary mask to COCO polygon representation
    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.
    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation 
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)

    return polygons

def create_annotation_info(annotation_id, image_id, category_info, 
                           binary_mask=None, image_size=None, tolerance=2, 
                           bounding_box=None, keypoints=None, num_keypoints=0):

    if binary_mask is not None:
        if image_size is not None:
            binary_mask = resize_binary_mask(binary_mask, image_size)

        binary_mask_encoded = mask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))

        area = mask.area(binary_mask_encoded)
        if area < 1:
            return None
        
        if bounding_box is None:
            bounding_box = mask.toBbox(binary_mask_encoded)
    else:
        area = 0
        segmentation = []
    
    if bounding_box is None:
        bounding_box = []

    annotation_info = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_info,
        "iscrowd": 0,
        "area": area,
        "bbox": bounding_box,
        "segmentation": segmentation,
        "num_keypoints": num_keypoints,
        "keypoints": keypoints,
    } 

    return annotation_info

def create_image_info(file_name, img_size, image_id=1,
                      data_captured=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')):
    image_info = {
        "license": 1,
        "file_name": file_name,
        "height": img_size[1],
        "width": img_size[0],
        "data_captured": data_captured,
        "coco_url": "",
        "flickr_url": "",
        "id": 1
    }
    return image_info

def create_annotation_dict(image_info, annotations):
    categroies = [{
        "supercategory":"person",
        "id":1,
        "name":"person",
        "keypoints":["nose","left_eye","right_eye","left_ear","right_ear","left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist","left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"],
        "skeleton":[[16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],[6,8],[7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]
    }]

    dict_result = {
            "images": [image_info],
            "annotations": annotations,
            "categories":categroies
        }
    return dict_result


def drawAnns(img, anns, cats):
    """
    Display the specified annotations.
    :param anns (array of object): annotations to display
    :return: None
    """
    if len(anns) == 0:
        return 0
    color = []
    for ann in anns:
        # c = (255*np.random.random((1, 3))*0.5+0.5).tolist()[0]
        c = (60, 20, 220)
        if 'bbox' in ann and type(ann['bbox']) == list:
            l,t,w,h = np.array(ann['bbox']).astype(int)
            cv2.rectangle(img, (l,t), (l+w,t+h), (0,255,0), 1)

        if 'keypoints' in ann and type(ann['keypoints']) == list:
            # turn skeleton into zero-based index
            sks = np.array(cats['skeleton'])-1
            kp = np.array(ann['keypoints'])
            x = kp[0::3].astype(int)
            y = kp[1::3].astype(int)
            v = kp[2::3].astype(int)
            for idx, sk in enumerate(sks[:-2]):
                if np.all(v[sk]>0) and idx != 12:
                    p1,p2 = sk
                    cv2.line(img, (x[p1], y[p1]), (x[p2], y[p2]), (50,50,50), thickness=2, lineType=1)
                    cv2.line(img, (x[p1], y[p1]), (x[p2], y[p2]), c, thickness=1, lineType=1)
            ### additional line
            if v[5] != 0 and v[6] != 0 and v[0] != 0:
                center_shoulder = ((x[5]+x[6])//2, (y[5]+y[6])//2)
                nose = (x[0],y[0])
                cv2.line(img, center_shoulder, nose, c, thickness=2)
            for idx, data in enumerate(zip(x,y,v,cats['keypoints'])):
                x,y,v,cat = data
                if idx%2 == 0: c = (50,220,220) # r : yellow
                else: c = (60, 20, 220)
                if v > 0:
                    cv2.circle(img, (x,y), 4, c, thickness=-1)
                if v > 1:
                    cv2.circle(img, (x,y), 5, (0,0,0), thickness= 1, lineType=1)
                # print ()
                # cv2.imshow('show', img)
                # key = cv2.waitKey()
                # if key == 27: exit()

    return img

if __name__ == '__main__':
    anno = create_annotation_info(12111, 10000000, 1, keypoints=[1,2,3,1,2,3], num_keypoints=2)

    cv2.namedWindow('show', 0)
    cv2.resizeWindow('show', 800, 800)

    print (anno)

    coco = COCO('../../data/coco/annotations/person_keypoints_val2017.json')
    cats = coco.loadCats(coco.getCatIds())[0]
    print (cats)
    img_info = coco.loadImgs(324158)[0]
    img = cv2.imread('../../data/coco/images/val2017/%s'%img_info['file_name'], 1)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    anns = coco.loadAnns(coco.getAnnIds(imgIds=img_info['id']))
    print (anns)

    
    show = drawAnns(img, anns, cats)
    cv2.imshow('show', show)
    cv2.waitKey()

