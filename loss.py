import math
import torch
from torch import nn
from utils import *


class MultiBoxLoss(nn.Module):
    """
    The MultiBox loss, a loss function for object detection.
    This is a combination of:
    (1) a localization loss for the predicted locations of the boxes, and
    (2) a confidence loss for the predicted class scores.
    """

    def __init__(self, priors_cxcy, device, threshold=0.5, neg_pos_ratio=3, alpha=1.0):
        super(MultiBoxLoss, self).__init__()
        self.priors_cxcy = priors_cxcy
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha
        self.device = device

        # self.smooth_l1 = nn.L1Loss()
        self.ciou_loss = CIoULoss(reduction='mean')
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')

    def forward(self, predicted_locs, predicted_scores, boxes, labels):
        """
        Forward propagation.
        :param predicted_locs: predicted locations/boxes w.r.t the 8732 prior boxes, a tensor of dimensions (N, 8732, 4)
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 8732, n_classes)
        :param boxes: true  object bounding boxes in boundary coordinates, a list of N tensors
        :param labels: true object labels, a list of N tensors
        :return: multibox loss, a scalar
        """
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        n_classes = predicted_scores.size(2)

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        # true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(self.device)  # (N, 8732, 4)
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(self.device)  # (N, 8732)
        decoded_pred_boxes = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(self.device)
        decoded_true_boxes = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(self.device)

        # For each image
        for i in range(batch_size):
            n_objects = boxes[i].size(0)

            overlap = find_jaccard_overlap(boxes[i], self.priors_xy)  # (n_objects, 8732)

            # For each prior, find the object that has the maximum overlap
            overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)  # (8732)

            # We don't want a situation where an object is not represented in our positive (non-background) priors -
            # 1. An object might not be the best object for all priors, and is therefore not in object_for_each_prior.
            # 2. All priors with the object may be assigned as background based on the threshold (0.5).

            # To remedy this -
            # First, find the prior that has the maximum overlap for each object.
            _, prior_for_each_object = overlap.max(dim=1)  # (N_o)

            # Then, assign each object to the corresponding maximum-overlap-prior. (This fixes 1.)
            object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(n_objects)).to(self.device)

            # To ensure these priors qualify, artificially give them an overlap of greater than 0.5. (This fixes 2.)
            overlap_for_each_prior[prior_for_each_object] = 1.

            # Labels for each prior
            label_for_each_prior = labels[i][object_for_each_prior]  # (8732)
            # Set priors whose overlaps with objects are less than the threshold to be background (no object)
            label_for_each_prior[overlap_for_each_prior < self.threshold] = 0  # (8732)

            # Store
            true_classes[i] = label_for_each_prior

            # Encode center-size object coordinates into the form we regressed predicted boxes to
            # true_locs[i] = cxcy_to_gcxgcy(xy_to_cxcy(boxes[i][object_for_each_prior]), self.priors_cxcy)  # (8732, 4)

            # Decode offsets into center-size coordinates and then boundary coordinates
            decoded_pred_boxes[i] = cxcy_to_xy(gcxgcy_to_cxcy(predicted_locs[i], self.priors_cxcy))
            decoded_true_boxes[i] = boxes[i][object_for_each_prior]

        # Identify priors that are positive (object/non-background)
        positive_priors = true_classes != 0  # (N, 8732)

        # LOCALIZATION LOSS
        # Localization loss is computed only over positive (non-background) priors
        # loc_loss = self.smooth_l1(predicted_locs[positive_priors], true_locs[positive_priors])  # (), scalar
        loc_loss = self.ciou_loss(decoded_pred_boxes[positive_priors], decoded_true_boxes[positive_priors])

        # Note: indexing with a torch.uint8 (byte) tensor flattens the tensor when indexing is across multiple dimensions (N & 8732)
        # So, if predicted_locs has the shape (N, 8732, 4), predicted_locs[positive_priors] will have (total positives, 4)

        # CONFIDENCE LOSS
        # Confidence loss is computed over positive priors and the most difficult negative priors in each image
        # Number of positive and hard-negative priors per image
        n_positives = positive_priors.sum(dim=1)  # (N)
        n_hard_negatives = self.neg_pos_ratio * n_positives  # (N)

        # First, find the loss for all priors
        conf_loss_all = self.cross_entropy(predicted_scores.view(-1, n_classes), true_classes.view(-1))  # (N * 8732)
        conf_loss_all = conf_loss_all.view(batch_size, n_priors)  # (N, 8732)

        # We already know which priors are positive
        conf_loss_pos = conf_loss_all[positive_priors]  # (sum(n_positives))

        # Next, find which priors are hard-negative
        # To do this, sort ONLY negative priors in each image in order of decreasing loss and take top n_hard_negatives
        conf_loss_neg = conf_loss_all.clone()  # (N, 8732)
        conf_loss_neg[positive_priors] = 0.  # (N, 8732), positive priors are ignored (never in top n_hard_negatives)
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)  # (N, 8732), sorted by decreasing hardness
        hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(self.device)  # (N, 8732)
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)  # (N, 8732)
        conf_loss_hard_neg = conf_loss_neg[hard_negatives]  # (sum(n_hard_negatives))

        # As in the paper, averaged over positive priors only, although computed over both positive and hard-negative priors
        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / n_positives.sum().float()  # (), scalar

        # TOTAL LOSS
        return conf_loss + self.alpha * loc_loss


class CIoULoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(CIoULoss, self).__init__()
        assert reduction in ['mean', 'sum', 'none']
        self.reduction = reduction

    def forward(self, boxes1, boxes2):
        """
        :param boxes1: in (xmin, ymin, xmax, ymax) coordinates
        :param boxes2: in (xmin, ymin, xmax, ymax) coordinates
        """
        w1 = boxes1[:, 2] - boxes1[:, 0]
        h1 = boxes1[:, 3] - boxes1[:, 1]
        w2 = boxes2[:, 2] - boxes2[:, 0]
        h2 = boxes2[:, 3] - boxes2[:, 1]

        area1 = w1 * h1
        area2 = w2 * h2

        center_x1 = (boxes1[:, 2] + boxes1[:, 0]) / 2
        center_y1 = (boxes1[:, 3] + boxes1[:, 1]) / 2
        center_x2 = (boxes2[:, 2] + boxes2[:, 0]) / 2
        center_y2 = (boxes2[:, 3] + boxes2[:, 1]) / 2

        inner_max_xy = torch.min(boxes1[:, 2:], boxes2[:, 2:])
        inner_min_xy = torch.max(boxes1[:, :2], boxes2[:, :2])
        outer_max_xy = torch.max(boxes1[:, 2:], boxes2[:, 2:])
        outer_min_xy = torch.min(boxes1[:, :2], boxes2[:, :2])

        inner_wh = torch.clamp((inner_max_xy - inner_min_xy), min=0)
        outer_wh = torch.clamp((outer_max_xy - outer_min_xy), min=0)

        inner_diag = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2
        outer_diag = (outer_wh[:, 0] ** 2) + (outer_wh[:, 1] ** 2)

        inner = inner_wh[:, 0] * inner_wh[:, 1]
        union = area1 + area2 - inner
        iou = inner / union

        v = 4 / (math.pi ** 2) * torch.pow((torch.atan(w2 / h2) - torch.atan(w1 / h1)), 2)
        with torch.no_grad():
            alpha = v / (1 - iou + v)
        loss = 1 - iou + inner_diag / outer_diag + alpha * v

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss


if __name__ == '__main__':
    boxes1 = torch.FloatTensor([0.1, 0.5, 0.7, 0.8]).repeat(4, 1)
    boxes2 = torch.FloatTensor([0.2, 0.2, 0.9, 0.7]).repeat(4, 1)
    print(boxes1.shape)
    print(boxes2.shape)
    loss_fn = CIoULoss(reduction='sum')
    ls = loss_fn(boxes1, boxes2)
    print(ls)
