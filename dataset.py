import os
import xml.etree.ElementTree as ET
import torch
from PIL import Image
from torch.utils.data import Dataset
from config import label_map


class VOCDataset(Dataset):
    def __init__(self, root, image_set, transform=None, keep_difficult=False):
        """
        VOC detection dataset.

        Args:
            root (string): root directory that holds 'Annotations', 'ImageSets', 'JPEGImages'
            image_set (string): must be in ['train', 'trainval', 'test']
            transform (callable, optional): transformation on the image, bboxes and labels
        """
        self.root = root
        self.image_set = image_set
        self.transform = transform
        self.keep_difficult = keep_difficult

        self.jpeg_image_path = os.path.join(root, 'JPEGImages', '%s.jpg')
        self.image_sets_path = os.path.join(root, 'ImageSets', 'Main', '%s.txt')
        self.annotation_path = os.path.join(root, 'Annotations', '%s.xml')

        with open(self.image_sets_path % self.image_set) as f:
            self.image_names = f.readlines()
        self.image_names = [n.strip('\n') for n in self.image_names]

    def __getitem__(self, item):
        # obtain the image
        image_name = self.image_names[item]
        image = Image.open(self.jpeg_image_path % image_name).convert('RGB')
        annotation = ET.parse(self.annotation_path % image_name).getroot()

        # obtain bboxes, labels and difficulties
        bboxes = []
        labels = []
        difficulties = []

        for obj in annotation.iter('object'):
            difficulty = int(obj.find('difficult').text)
            if not self.keep_difficult and difficulty == 1:
                continue

            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text) - 1
            ymin = int(bbox.find('ymin').text) - 1
            xmax = int(bbox.find('xmax').text) - 1
            ymax = int(bbox.find('ymax').text) - 1

            bboxes.append([xmin, ymin, xmax, ymax])
            labels.append(label_map[name])
            difficulties.append(difficulty)

        # convert bboxes, labels and difficulties to tensor
        bboxes = torch.FloatTensor(bboxes)
        labels = torch.LongTensor(labels)
        difficulties = torch.ByteTensor(difficulties)

        # transform the image, bboxes, labels and difficulties
        if self.transform is not None:
            image, bboxes, labels, difficulties = self.transform(image, bboxes, labels, difficulties)

        return image, bboxes, labels, difficulties

    def __len__(self):
        return len(self.image_names)


def collate_fn(batch):
    """
    This function returns:
        images: a tensor with shape (N, 3, 300, 300)
        bboxes: a list of N tensors
        labels: a list of N tensors
        difficulties: a list of N tensors
    """
    images = []
    bboxes = []
    labels = []
    difficulties = []

    for b in batch:
        images.append(b[0])
        bboxes.append(b[1])
        labels.append(b[2])
        difficulties.append(b[3])

    images = torch.stack(images, dim=0)

    return images, bboxes, labels, difficulties


'''
def show_image(image, boxes, labels, difficulties):
    image = image.permute(1, 2, 0).numpy()
    image = (image * 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    boxes = boxes.numpy()
    labels = labels.numpy()
    difficulties = difficulties.numpy()

    for i, box in enumerate(boxes):
        color = distinct_colors[labels[i]]
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        color = (b, g, r)
        thickness = 2
        pt1 = (box[0], box[1])
        pt2 = (box[2], box[3])
        image = cv2.rectangle(image, pt1, pt2, color=color, thickness=thickness)

        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5

        text = str(rev_label_map[labels[i]])
        (w, h), _ = cv2.getTextSize(text, font_face, font_scale, thickness)
        org = (box[0], box[1] + h)
        image = cv2.putText(image, text, org, font_face, font_scale, color, thickness)

        text = str(difficulties[i])
        (w, h), _ = cv2.getTextSize(text, font_face, font_scale, thickness)
        org = (box[2] - w, box[3])
        image = cv2.putText(image, text, org, font_face, font_scale, color, thickness)

    cv2.imshow('image', image)
    cv2.waitKey()


def main():
    transform = Transform(size=(600, 800), train=True)
    train_dataset = VOCDataset(root='data',
                               image_set='trainval',
                               transform=transform,
                               keep_difficult=True)
    it = iter(train_dataset)
    for _ in range(20):
        next(it)
    for _ in range(20):
        images, bboxes, labels, difficulties = next(it)
        show_image(images, bboxes, labels, difficulties)


if __name__ == '__main__':
    main()
'''
