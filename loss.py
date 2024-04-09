import torch
from torch import nn


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
