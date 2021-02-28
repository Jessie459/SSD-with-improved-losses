import argparse
import os
import shutil

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import VOCDataset, collate_fn
from model import SSD300
from loss import MultiBoxLoss
from transform import Transform
from utils import set_seed, AverageMeter, adjust_learning_rate
from config import label_map

parser = argparse.ArgumentParser(description='Train Single Shot MultiBox Detector')
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--data_root', type=str, default='data')
parser.add_argument('--save_root', type=str, default='weights')
parser.add_argument('--image_set', type=str, default='train')
parser.add_argument('--save_freq', type=int, default=10)
parser.add_argument('--print_freq', type=int, default=50)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--num_workers', type=int, default=0)
args = parser.parse_args()

device = torch.device(args.device)


def train():
    set_seed(seed=10)
    os.makedirs(args.save_root, exist_ok=True)

    # create model, optimizer and criterion
    model = SSD300(n_classes=len(label_map), device=device)
    biases = []
    not_biases = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if name.endswith('.bias'):
                biases.append(param)
            else:
                not_biases.append(param)
    model = model.to(device)
    optimizer = torch.optim.SGD(
        params=[{'params': biases, 'lr': 2 * args.lr},
                {'params': not_biases}],
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    if args.resume is None:
        start_epoch = 0
    else:
        checkpoint = torch.load(args.resume, map_location=device)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    print(f'Training will start at epoch {start_epoch}.')

    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy, device=device, alpha=args.alpha)
    criterion = criterion.to(device)

    '''
    scheduler = StepLR(optimizer=optimizer,
                       step_size=20,
                       gamma=0.5,
                       last_epoch=start_epoch - 1,
                       verbose=True)
    '''

    # load data
    transform = Transform(size=(300, 300), train=True)
    train_dataset = VOCDataset(root=args.data_root,
                               image_set=args.image_set,
                               transform=transform,
                               keep_difficult=True)
    train_loader = DataLoader(dataset=train_dataset,
                              collate_fn=collate_fn,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=True,
                              pin_memory=True)

    losses = AverageMeter()
    for epoch in range(start_epoch, args.num_epochs):
        # decay learning rate at particular epochs
        if epoch in [120, 140, 160]:
            adjust_learning_rate(optimizer, 0.1)

        # train model
        model.train()
        losses.reset()
        bar = tqdm(train_loader, desc='Train the model')
        for i, (images, bboxes, labels, _) in enumerate(bar):
            images = images.to(device)
            bboxes = [b.to(device) for b in bboxes]
            labels = [l.to(device) for l in labels]

            predicted_bboxes, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, num_classes)
            loss = criterion(predicted_bboxes, predicted_scores, bboxes, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item(), images.size(0))

            if i % args.print_freq == args.print_freq - 1:
                bar.write(f'Average Loss: {losses.avg:.4f}')

        bar.write(f'Epoch: [{epoch + 1}|{args.num_epochs}] '
                  f'Average Loss: {losses.avg:.4f}')
        # adjust learning rate
        # scheduler.step()

        # save model
        state_dict = {'epoch': epoch,
                      'model': model.state_dict(),
                      'optimizer': optimizer.state_dict()}
        save_path = os.path.join(args.save_root, 'ssd300.pth')
        torch.save(state_dict, save_path)

        if epoch % args.save_freq == args.save_freq - 1:
            shutil.copyfile(save_path, os.path.join(args.save_root, f'ssd300_epochs_{epoch + 1}.pth'))


if __name__ == '__main__':
    train()
