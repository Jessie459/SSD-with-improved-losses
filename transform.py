import random
import torch
import torchvision.transforms.functional as TF
from utils import find_jaccard_overlap


class Transform:
    def __init__(self, size=(300, 300), train=True):
        self.size = size
        self.train = train

    def __call__(self, image, bboxes, labels, difficulties):
        """
        Args:
            image: a PIL image
            bboxes: a tensor with shape (num_objects, 4)
            labels: a tensor with shape (num_objects)
            difficulties: a tensor with shape (num_objects)
        """
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        new_image = image
        new_bboxes = bboxes
        new_labels = labels
        new_difficulties = difficulties

        if self.train:
            new_image = photometric_distort(new_image)
            new_image = TF.to_tensor(new_image)
            if random.random() < 0.5:
                new_image, new_bboxes = expand(new_image, new_bboxes, fill=mean)
            new_image, new_bboxes, new_labels, new_difficulties = random_crop(
                new_image, new_bboxes, new_labels, new_difficulties
            )
            new_image = TF.to_pil_image(new_image)
            if random.random() < 0.5:
                new_image, new_bboxes = hflip(new_image, new_bboxes)

        new_image, new_bboxes = resize(new_image, new_bboxes, size=self.size)
        new_image = TF.to_tensor(new_image)
        new_image = TF.normalize(new_image, mean=mean, std=std)

        return new_image, new_bboxes, new_labels, new_difficulties


def photometric_distort(image):
    """
    Distort brightness, contrast, saturation, hue in random order,
    each with a 0.5 probability.

    :param image: a PIL image
    """
    new_image = image

    distortions = [
        TF.adjust_brightness,
        TF.adjust_contrast,
        TF.adjust_saturation,
        TF.adjust_hue
    ]
    random.shuffle(distortions)

    for distortion in distortions:
        if random.random() < 0.5:
            if distortion.__name__ == 'adjust_hue':
                adjust_factor = random.uniform(-0.1, 0.1)
            else:
                adjust_factor = random.uniform(0.5, 1.5)
            new_image = distortion(new_image, adjust_factor)

    return new_image


def expand(image, bboxes, fill):
    """
    Perform a zooming out operation by placing the image in a larger canvas
    to help detect smaller objects.

    :param image: a tensor with shape (C, H, W)
    :param bboxes: a tensor with shape (num_objects, 4)
    :param fill: a sequence containing three items
    """
    org_h = image.shape[1]
    org_w = image.shape[2]
    scale = random.uniform(1, 4)
    new_h = int(org_h * scale)
    new_w = int(org_w * scale)

    # create the canvas with fill value
    fill = torch.FloatTensor(fill).unsqueeze(1).unsqueeze(1)
    new_image = torch.ones((3, new_h, new_w), dtype=torch.float32) * fill

    # place the image on the canvas
    ymin = random.randint(0, new_h - org_h)
    ymax = ymin + org_h
    xmin = random.randint(0, new_w - org_w)
    xmax = xmin + org_w
    new_image[:, ymin:ymax, xmin:xmax] = image

    # adjust bounding bboxes' coordinates
    new_bboxes = bboxes + torch.FloatTensor([xmin, ymin, xmin, ymin]).unsqueeze(0)

    return new_image, new_bboxes


def random_crop(image, bboxes, labels, difficulties):
    """
    Performs a random crop to help detect larger and partial objects.

    :param image: a tensor with shape (C, H, W)
    :param bboxes: a tensor with shape (num_objects, 4)
    :param labels: a tensor with shape (num_objects)
    :param difficulties: a tensor with shape (num_objects)
    """
    org_h = image.shape[1]
    org_w = image.shape[2]

    # keep choosing a minimum overlap until a successful crop is made
    while True:
        min_overlap = random.choice([0.0, 0.1, 0.3, 0.5, 0.7, 0.9, None])
        if min_overlap is None:
            return image, bboxes, labels, difficulties

        # try up to 50 times for this choice of minimum overlap
        max_trials = 50
        for _ in range(max_trials):
            scale_h = random.uniform(0.3, 1)
            scale_w = random.uniform(0.3, 1)
            new_h = int(scale_h * org_h)
            new_w = int(scale_w * org_w)

            # aspect ratio must be in [0.5, 2]
            aspect_ratio = new_h / new_w
            if not 0.5 < aspect_ratio < 2:
                continue

            # calculate crop coordinates
            ymin = random.randint(0, org_h - new_h)
            ymax = ymin + new_h
            xmin = random.randint(0, org_w - new_w)
            xmax = xmin + new_w
            crop = torch.FloatTensor([xmin, ymin, xmax, ymax])
            overlap = find_jaccard_overlap(crop.unsqueeze(0), bboxes).squeeze(0)  # (num_objects)
            if overlap.max().item() < min_overlap:
                continue

            # crop image
            new_image = image[:, ymin:ymax, xmin:xmax]  # (3, new_h, new_w)

            # find centers of original bounding bboxes
            centers = (bboxes[:, :2] + bboxes[:, 2:]) / 2  # (num_objects, 2)
            centers_in_crop = (centers[:, 0] > xmin) * (centers[:, 0] < xmax) * \
                              (centers[:, 1] > ymin) * (centers[:, 1] < ymax)
            if not centers_in_crop.any():
                continue

            # filter bounding bboxes and labels
            new_bboxes = bboxes[centers_in_crop, :]
            new_labels = labels[centers_in_crop]
            new_difficulties = difficulties[centers_in_crop]

            # calculate bounding bboxes' new coordinates in the crop
            new_bboxes[:, :2] = torch.max(new_bboxes[:, :2], crop[:2])
            new_bboxes[:, :2] -= crop[:2]
            new_bboxes[:, 2:] = torch.min(new_bboxes[:, 2:], crop[2:])
            new_bboxes[:, 2:] -= crop[:2]

            return new_image, new_bboxes, new_labels, new_difficulties


def hflip(image, bboxes):
    """
    Flip image horizontally.

    :param image: a PIL image
    :param bboxes: a tensor with shape (num_objects, 4)
    """
    # flip the image horizontally
    new_image = TF.hflip(image)

    # flip bboxes horizontally
    new_bboxes = bboxes
    new_bboxes[:, 0] = image.width - bboxes[:, 0] - 1  # xmin -> new xmax
    new_bboxes[:, 2] = image.width - bboxes[:, 2] - 1  # xmax -> new xmin
    new_bboxes = new_bboxes[:, [2, 1, 0, 3]]

    return new_image, new_bboxes


def resize(image, bboxes, size=(300, 300)):
    """
    Resize image. For the SSD 300, resize to (300, 300)

    :param image: a PIL image
    :param bboxes: a tensor with shape (num_objects, 4)
    :param size: a sequence like (h, w) or an int
    """
    # resize the image
    new_image = TF.resize(image, size)

    # resize bounding bboxes
    w = image.width
    h = image.height
    org_size = torch.FloatTensor([w, h, w, h]).unsqueeze(0)
    new_bboxes = bboxes / org_size

    return new_image, new_bboxes
