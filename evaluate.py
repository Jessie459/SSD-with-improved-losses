import argparse
import os
from pprint import PrettyPrinter
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import VOCDataset, collate_fn
from model import SSD300
from transform import Transform
from utils import label_map, calculate_mAP

parser = argparse.ArgumentParser(description='Evaluate Single Shot MultiBox Detector')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--data_root', type=str, default='data')
parser.add_argument('--model_root', type=str, default='weights')
parser.add_argument('--model_name', type=str, default='ssd300.pth')
parser.add_argument('--image_set', type=str, default='test')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--num_workers', type=int, default=0)
args = parser.parse_args()

device = torch.device(args.device)


def evaluate():
    checkpoint_path = os.path.join(args.model_root, args.model_name)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = SSD300(n_classes=len(label_map), device=device).to(device)
    model.load_state_dict(checkpoint['model'])

    transform = Transform(size=(300, 300), train=False)
    test_dataset = VOCDataset(
        root=args.data_root,
        image_set=args.image_set,
        transform=transform,
        keep_difficult=True
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True
    )

    detected_bboxes = []
    detected_labels = []
    detected_scores = []
    true_bboxes = []
    true_labels = []
    true_difficulties = []

    model.eval()
    with torch.no_grad():
        bar = tqdm(test_loader, desc='Evaluate the model')
        for i, (images, bboxes, labels, difficulties) in enumerate(bar):
            images = images.to(device)
            bboxes = [b.to(device) for b in bboxes]
            labels = [l.to(device) for l in labels]
            difficulties = [d.to(device) for d in difficulties]

            predicted_bboxes, predicted_scores = model(images)
            _bboxes, _labels, _scores = model.detect_objects(
                predicted_bboxes,
                predicted_scores,
                min_score=0.01,
                max_overlap=0.45,
                top_k=200
            )

            detected_bboxes += _bboxes
            detected_labels += _labels
            detected_scores += _scores
            true_bboxes += bboxes
            true_labels += labels
            true_difficulties += difficulties

        all_ap, mean_ap = calculate_mAP(
            detected_bboxes,
            detected_labels,
            detected_scores,
            true_bboxes,
            true_labels,
            true_difficulties,
            device=device
        )

    pretty_printer = PrettyPrinter()
    pretty_printer.pprint(all_ap)
    print('Mean Average Precision (mAP): %.4f' % mean_ap)


if __name__ == '__main__':
    evaluate()
