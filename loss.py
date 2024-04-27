import torch
from torch import nn
from torch.autograd import Variable

import config


class DetectionClassificationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bin_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.class_loss = nn.CrossEntropyLoss()

    def forward(self, y_pred, y):
        y_positive_mask = (y != 0).to(dtype=torch.float32, device=y.device)

        y_bin_logit = y_pred[:, 1]
        y_class_logit = y_pred[:, 1:]

        detection_bin_crossentropy_loss = self.bin_loss(y_bin_logit, y_positive_mask)
        detection_positive_loss = (detection_bin_crossentropy_loss * y_positive_mask).mean()
        detection_negative_loss = (detection_bin_crossentropy_loss * (1 - y_positive_mask))
        positive_pixels_count = max(torch.sum(y_positive_mask, dtype=torch.int).item(), 1)
        negative_pixels_count = max(torch.sum((1 - y_positive_mask), dtype=torch.int).item(), 1)
        detection_k_worst_negative_loss, _ = (detection_negative_loss.view(-1).
                                              topk(k=min(positive_pixels_count, negative_pixels_count)))
        detection_k_worst_negative_loss = detection_k_worst_negative_loss.mean()
        detection_negative_loss = detection_negative_loss.mean()

        weight_positive, weight_negative, weight_k_worst_negative = 15, 1, 5
        detection_loss = \
            weight_positive * detection_positive_loss + \
            weight_negative * detection_negative_loss + \
            weight_k_worst_negative * detection_k_worst_negative_loss

        target = ((y - 1) * y_positive_mask).to(dtype=torch.long)
        class_loss = self.class_loss(y_class_logit * y_positive_mask, target)

        alpha = 1
        total_loss = detection_loss + alpha * class_loss

        return total_loss


class DiscriminativeLoss(nn.Module):
    def __init__(self, delta_var, delta_dist, norm,
                 size_average=True, reduce=True):
        super(DiscriminativeLoss, self).__init__()
        self.reduce = reduce

        self.delta_var = float(delta_var)
        self.delta_dist = float(delta_dist)
        self.norm = int(norm)

        self.alpha = 1.0
        self.beta = 1.0
        self.gamma = 0.001

        assert self.norm in [1, 2]

    def forward(self, prediction, target, n_objects):
        """
        input: bs, n_dims, height, width
        target: bs, n_instances, height, width
        n_objects: bs
        """

        bs, n_dims, height, width = prediction.shape
        n_instances = target.shape[1]

        prediction = prediction.permute(0, 2, 3, 1).contiguous().view(bs, height * width, n_dims)
        target = target.permute(0, 2, 3, 1).contiguous().view(bs, height * width, n_instances)

        cluster_means = calculate_means(prediction, target, n_objects)
        var_term = calculate_variance_term(prediction, target, cluster_means, n_objects, self.delta_var, self.norm)
        dist_term = calculate_distance_term(cluster_means, n_objects, self.delta_dist, self.norm)
        reg_term = calculate_regularization_term(cluster_means, n_objects, self.norm)

        loss = self.alpha * var_term + self.beta * dist_term + self.gamma * reg_term

        return loss


def calculate_means(prediction, gt, n_objects, not_from_loss=False):
    """
    prediction: bs, height * width, n_dims
    gt: bs, height * width, n_instances  # one-hot
    n_objects: bs
    """

    if not_from_loss:
        bs, n_dims, height, width = prediction.shape
        n_instances = gt.shape[1]

        prediction = prediction.permute(0, 2, 3, 1).contiguous().view(bs, height * width, n_dims)
        gt = gt.permute(0, 2, 3, 1).contiguous().view(bs, height * width, n_instances)

    bs, n_loc, n_dims = prediction.shape
    n_instances = gt.shape[2]

    # bs, n_loc, n_instances, n_dims
    pred_repeated = prediction.unsqueeze(2).expand(bs, n_loc, n_instances, n_dims)
    # bs, n_loc, n_instances, 1
    gt_expanded = gt.unsqueeze(3)

    pred_masked = pred_repeated * gt_expanded

    means = []
    for i in range(bs):
        n_objects_sample = n_objects[i]   # = n_instances for semantic segmentation
        # n_loc, n_objects, n_dims
        pred_masked_sample = pred_masked[i, :, :n_objects_sample]
        # n_loc, n_objects, 1
        gt_expanded_sample = gt_expanded[i, :, :n_objects_sample]

        mean_sample = pred_masked_sample.sum(0) / gt_expanded_sample.sum(0)  # n_objects, n_dims
        if (n_instances - n_objects_sample) != 0:
            n_fill_objects = int(n_instances - n_objects_sample)
            fill_sample = torch.zeros(n_fill_objects, n_dims, device=config.Settings.device, requires_grad=True)
            mean_sample = torch.cat((mean_sample, fill_sample), dim=0)
        means.append(mean_sample)

    # bs, n_instances, n_dims
    means = torch.stack(means)

    return means


def calculate_variance_term(prediction, gt, means, n_objects, delta_v, norm=2):
    """
    pred: bs, height * width, n_dims
    gt: bs, height * width, n_instances
    means: bs, n_instances, n_dims
    """

    bs, n_loc, n_dims = prediction.shape
    n_instances = gt.shape[2]

    # bs, n_loc, n_instances, n_dims
    means = means.unsqueeze(1).expand(bs, n_loc, n_instances, n_dims)
    # bs, n_loc, n_instances, n_dims
    prediction = prediction.unsqueeze(2).expand(bs, n_loc, n_instances, n_dims)
    # bs, n_loc, n_instances, n_dims
    gt = gt.unsqueeze(3).expand(bs, n_loc, n_instances, n_dims)

    var = (torch.clamp(torch.norm((prediction - means), norm, 3) - delta_v, min=0.0) ** 2) * gt[:, :, :, 0]

    var_term = 0.0
    for i in range(bs):
        var_sample = var[i, :, :n_objects[i]]  # n_loc, n_objects
        gt_sample = gt[i, :, :n_objects[i], 0]  # n_loc, n_objects

        var_term += torch.sum(var_sample) / torch.sum(gt_sample)
    var_term = var_term / bs

    return var_term


def calculate_distance_term(means, n_objects, delta_d, norm=2):
    """means: bs, n_instances, n_dims"""

    bs, n_instances, n_filters = means.shape

    dist_term = 0.0
    for i in range(bs):
        n_objects_sample = int(n_objects[i])

        if n_objects_sample <= 1:
            continue

        mean_sample = means[i, : n_objects_sample, :]  # n_objects, n_filters
        means_1 = mean_sample.unsqueeze(1).expand(n_objects_sample, n_objects_sample, n_filters)
        means_2 = means_1.permute(1, 0, 2)

        diff = means_1 - means_2  # n_objects, n_objects, n_filters

        _norm = torch.norm(diff, norm, 2)

        margin = 2 * delta_d * (1.0 - torch.eye(n_objects_sample, device=config.Settings.device, requires_grad=True))

        dist_term_sample = torch.sum(torch.clamp(margin - _norm, min=0.0) ** 2)
        dist_term_sample = dist_term_sample / (n_objects_sample * (n_objects_sample - 1))
        dist_term += dist_term_sample

    dist_term = dist_term / bs

    return dist_term


def calculate_regularization_term(means, n_objects, norm):
    """
    means: bs, n_instances, n_dims
    """

    bs, n_instances, n_filters = means.shape

    reg_term = 0.0
    for i in range(bs):
        mean_sample = means[i, : n_objects[i], :]  # n_objects, n_dims
        _norm = torch.norm(mean_sample, norm, 1)
        reg_term += torch.mean(_norm)
    reg_term = reg_term / bs

    return reg_term
