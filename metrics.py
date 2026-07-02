import numpy as np
import torch.nn.functional as F
import torch



def iou_score(output, target):
    smooth = 1e-5
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)


def dice_coef(output, target):
    smooth = 1e-5

    # we need to use sigmoid because the output of Unet is logit.
    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
           (output.sum() + target.sum() + smooth)

def precision_coef(output, target):
    smooth = 1e-5
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_).sum()
    return (intersection + smooth) / (union + smooth)

def recall_coef(output, target):
    smooth = 1e-5
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (target_).sum()
    return (intersection + smooth) / (union + smooth)


def calculate_mae(predicted_labels, true_labels):
    if torch.is_tensor(predicted_labels):
        output = torch.sigmoid(predicted_labels).data.cpu().numpy()
    if torch.is_tensor(true_labels):
        target = true_labels.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    mae = np.mean(np.abs(output_ ^ target_))
    return mae


def compute_f1_score(output, target, beta=1):
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    true_positive = ((output_ == 1) & (target_ == 1)).sum().item()
    false_positive = ((output_ == 1) & (target_ == 0)).sum().item()
    false_negative = ((output_ == 0) & (target_ == 1)).sum().item()

    denominator = true_positive + false_positive
    if denominator == 0:
        precision = 0  # 设置默认值为0，或者根据需求设置其他合适的值
    else:
        precision = true_positive / denominator

    recall = true_positive / (true_positive + false_negative)
    if (beta ** 2 * precision) + recall == 0:
        f1_score = 0  # 设置默认值为0
    else:
        f1_score = (1 + beta ** 2) * (precision * recall) / ((beta ** 2 * precision) + recall)

    return f1_score


def compute_f2_score(output, target, beta=2):
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    true_positive = ((output_ == 1) & (target_ == 1)).sum().item()
    false_positive = ((output_ == 1) & (target_ == 0)).sum().item()
    false_negative = ((output_ == 0) & (target_ == 1)).sum().item()

    denominator = true_positive + false_positive
    if denominator == 0:
        precision = 0  # 设置默认值为0，或者根据需求设置其他合适的值
    else:
        precision = true_positive / denominator

    recall = true_positive / (true_positive + false_negative)
    if (beta ** 2 * precision) + recall == 0:
        f2_score = 0  # 设置默认值为0
    else:
        f2_score = (1 + beta ** 2) * (precision * recall) / ((beta ** 2 * precision) + recall)

    return f2_score


def compute_f1_2_score(output, target, beta=0.5):
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    true_positive = ((output_ == 1) & (target_ == 1)).sum().item()
    false_positive = ((output_ == 1) & (target_ == 0)).sum().item()
    false_negative = ((output_ == 0) & (target_ == 1)).sum().item()

    denominator = true_positive + false_positive
    if denominator == 0:
        precision = 0  # 设置默认值为0，或者根据需求设置其他合适的值
    else:
        precision = true_positive / denominator

    recall = true_positive / (true_positive + false_negative)
    if (beta ** 2 * precision) + recall == 0:
        f1_2_score = 0  # 设置默认值为0
    else:
        f1_2_score = (1 + beta ** 2) * (precision * recall) / ((beta ** 2 * precision) + recall)

    return f1_2_score

