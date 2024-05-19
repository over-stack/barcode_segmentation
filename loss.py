import torch
from torch import nn

import config


class DetectionClassificationLoss(nn.Module):
    def __init__(self, weight_positive=15, weight_negative=1, weight_k_worst_negative=5, detection_loss_weight=1,
                 classification_loss_weight=1, class_weights=None):
        super().__init__()
        self.bin_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.class_loss = nn.CrossEntropyLoss(reduction='none', weight=class_weights)
        self.weight_positive = weight_positive
        self.weight_negative = weight_negative
        self.weight_k_worst_negative = weight_k_worst_negative
        self.detection_loss_weight = detection_loss_weight
        self.classification_loss_weight = classification_loss_weight

    def forward(self, y_pred: torch.Tensor, y: torch.Tensor):
        # prediction channels:
        # 0 - objects mask (inverse background)
        # 1 ... n - classes masks

        # target channels:
        # 0 ... n - classes - masks (0 - background mask)

        y_bin_logit = y_pred[:, 0]  # objects mask
        y_positive_mask = (y[:, 0] != 1).to(dtype=torch.float32, device=y.device)  # not background (target objs mask)

        positive_pixels_count = max(torch.sum(y_positive_mask, dtype=torch.int).item(), 1)
        negative_pixels_count = max(torch.sum((1 - y_positive_mask), dtype=torch.int).item(), 1)

        detection_bin_crossentropy_loss = self.bin_loss(y_bin_logit, y_positive_mask)

        detection_positive_loss_pixels = (detection_bin_crossentropy_loss * y_positive_mask)
        detection_positive_loss = detection_positive_loss_pixels.sum() / positive_pixels_count

        detection_negative_loss_pixels = (detection_bin_crossentropy_loss * (1 - y_positive_mask))
        detection_negative_loss = detection_negative_loss_pixels.sum() / negative_pixels_count

        detection_k_worst_negative_loss, _ = detection_negative_loss_pixels.view(-1).topk(k=positive_pixels_count)
        detection_k_worst_negative_loss = detection_k_worst_negative_loss.mean()

        detection_loss = (
            self.weight_positive * detection_positive_loss +
            self.weight_negative * detection_negative_loss +
            self.weight_k_worst_negative * detection_k_worst_negative_loss
        )

        class_loss_pixels = self.class_loss(y_pred[:, 1:], y[:, 1:].to(y_pred.dtype))
        class_loss = (class_loss_pixels * y_positive_mask).sum() / positive_pixels_count

        total_loss = self.detection_loss_weight * detection_loss + self.classification_loss_weight * class_loss

        return total_loss


class DiscriminativeLoss(nn.Module):
    def __init__(self, delta_var=0.5, delta_dist=1.5, norm=2, var_term_weight=1.0, dist_term_weight=1.0,
                 reg_term_weight=0.001, class_weights=None):
        super(DiscriminativeLoss, self).__init__()

        self.delta_var = delta_var
        self.delta_dist = delta_dist
        self.norm = norm

        self.var_term_weight = var_term_weight
        self.dist_term_weight = dist_term_weight
        self.reg_term_weight = reg_term_weight

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

        loss = self.var_term_weight * var_term + self.dist_term_weight * dist_term + self.reg_term_weight * reg_term

        return loss


def calculate_means(prediction, gt, n_objects, not_from_loss=False):
    """
    prediction: bs, height * width, n_dims
    gt: bs, height * width, n_instances
    n_objects: bs
    """

    if not_from_loss:
        # input: bs, n_dims, height, width
        # target: bs, n_instances, height, width
        # n_objects: bs

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

        gt_expanded_sample_sum = gt_expanded_sample.sum(0)
        mean_sample = pred_masked_sample.sum(0) / torch.clamp(gt_expanded_sample_sum, min=1)  # n_objects, n_dims
        if (n_instances - n_objects_sample) != 0:
            n_fill_objects = int(n_instances - n_objects_sample)
            fill_sample = torch.zeros((n_fill_objects, n_dims), device=config.Settings.device, requires_grad=True)
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

    # bs, n_loc, n_instances
    var = (torch.clamp(torch.norm((prediction - means), norm, 3) - delta_v, min=0.0) ** 2) * gt[:, :, :, 0]

    var_term = 0.0
    for i in range(bs):
        var_sample = var[i, :, :n_objects[i]]  # n_loc, n_objects
        gt_sample = gt[i, :, :n_objects[i], 0]  # n_loc, n_objects

        var_term += torch.sum(var_sample.sum(dim=0) / torch.clamp(gt_sample.sum(dim=0), min=1.0)) / n_objects[i]

    var_term = var_term / bs

    return var_term


def calculate_distance_term(means, n_objects, delta_d, norm=2):
    """means: bs, n_instances, n_dims"""

    bs, n_instances, n_dims = means.shape

    dist_term = 0.0
    for i in range(bs):
        if n_objects[i] <= 1:
            continue

        # n_objects, n_dims
        mean_sample = means[i, :n_objects[i]]
        # n_objects, n_objects, n_dims
        means_1 = mean_sample.unsqueeze(1).expand(n_objects[i], n_objects[i], n_dims)
        means_2 = means_1.permute(1, 0, 2)

        # n_objects, n_objects, n_dims
        diff = means_1 - means_2
        _norm = torch.norm(diff, norm, 2)
        margin = 2 * delta_d * (1.0 - torch.eye(n_objects[i], device=config.Settings.device, requires_grad=True))
        dist_term_sample = torch.sum(torch.clamp(margin - _norm, min=0.0) ** 2) / (n_objects[i] * (n_objects[i] - 1))
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
        # n_objects, n_dims
        mean_sample = means[i, :n_objects[i]]
        _norm = torch.norm(mean_sample, norm, 1)
        reg_term += torch.sum(_norm) / n_objects[i]
    reg_term = reg_term / bs

    return reg_term
