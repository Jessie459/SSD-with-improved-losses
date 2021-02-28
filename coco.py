import os
import shutil
from lxml import etree

import cv2
from pycocotools.coco import COCO


img_root = 'D:\\dataset\\train2014'
ann_path = 'D:\\dataset\\annotations_trainval2014\\instances_train2014.json'

img_save_root = 'coco\\images'
ann_save_root = 'coco\\annotations'

os.makedirs(img_save_root, exist_ok=True)
os.makedirs(ann_save_root, exist_ok=True)


def save_img_names(names, path):
    with open(path, 'a+') as f:
        for name in names:
            f.write(name + '\n')


def save_annotations(img_name, objs):
    ann_save_path = os.path.join(ann_save_root, img_name + '.xml')
    img_save_path = os.path.join(img_save_root, img_name + '.jpg')
    img_path = os.path.join(img_root, img_name + '.jpg')
    img = cv2.imread(img_path)

    shutil.copyfile(img_path, img_save_path)

    annotation = etree.Element('annotation')
    etree.SubElement(annotation, 'filename').text = img_name + '.jpg'

    size = etree.SubElement(annotation, 'size')
    etree.SubElement(size, 'height').text = str(img.shape[0])
    etree.SubElement(size, 'width').text = str(img.shape[1])
    etree.SubElement(size, 'depth').text = str(img.shape[2])

    for obj in objs:
        _object = etree.SubElement(annotation, 'object')
        etree.SubElement(_object, 'name').text = obj[0]
        etree.SubElement(_object, 'difficult').text = '0'

        bndbox = etree.SubElement(_object, 'bndbox')
        etree.SubElement(bndbox, 'xmin').text = str(obj[1])
        etree.SubElement(bndbox, 'ymin').text = str(obj[2])
        etree.SubElement(bndbox, 'xmax').text = str(obj[3])
        etree.SubElement(bndbox, 'ymax').text = str(obj[4])

    tree = etree.ElementTree(annotation)
    tree.write(ann_save_path, pretty_print=True)


def main():
    count = 0
    coco = COCO(ann_path)

    id2category = {}
    for cat in coco.dataset['categories']:
        id2category[cat['id']] = cat['name']

    img_names_train = []
    img_names_test = []
    cat_names = ['elephant']
    cat_ids = coco.getCatIds(catNms=cat_names)
    img_ids = coco.getImgIds(catIds=cat_ids)
    for img_id in img_ids:
        img = coco.loadImgs(img_id)[0]
        img_name = img['file_name'][:-4]

        ann_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids)
        if len(ann_ids) > 4:
            continue
        count += 1
        if count <= 200:
            img_names_train.append(img_name)
        elif count <= 400:
            img_names_test.append(img_name)
        else:
            break
        anns = coco.loadAnns(ann_ids)
        objs = []
        for ann in anns:
            if 'bbox' in ann.keys():
                bbox = ann['bbox']
                xmin = int(bbox[0])
                ymin = int(bbox[1])
                xmax = int(bbox[2] + bbox[0])
                ymax = int(bbox[3] + bbox[1])
                category = id2category[ann['category_id']]
                objs.append([category, xmin, ymin, xmax, ymax])

        save_annotations(img_name, objs)

    save_img_names(img_names_train, 'coco\\train.txt')
    save_img_names(img_names_test, 'coco\\test.txt')


if __name__ == '__main__':
    main()
