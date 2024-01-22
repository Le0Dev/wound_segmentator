def accuracy(output, target, threshold=0.5):
    """
    Calculate the accuracy between predicted and target values.

    Parameters:
    - output (torch.Tensor): Predicted values.
    - target (torch.Tensor): Ground truth values.
    - threshold (float): Threshold.

    Returns:
    - float: Accuracy value.
    """
    pred = (output > threshold).float()
    correct = (pred == target).float()
    accuracy = correct.sum() / (target.numel() + 1e-8)
    return accuracy.item()

def precision(output, target, threshold=0.5):
    """
    Calculate precision

    Parameters:
    - output (torch.Tensor): Predicted values.
    - target (torch.Tensor): Ground truth values.
    - threshold (float): Threshold.

    Returns:
    - float: Precision value.
    """
    pred = (output > threshold).float()
    true_positive = (pred * target).sum()
    false_positive = pred.sum() - true_positive
    precision = true_positive / (true_positive + false_positive + 1e-8)
    return precision.item()

def recall(output, target, threshold=0.5):
    """
    Calculate recall.

    Parameters:
    - output (torch.Tensor): Predicted values.
    - target (torch.Tensor): Ground truth values.
    - threshold (float): Threshold.

    Returns:
    - float: Recall value.
    """
    pred = (output > threshold).float()
    true_positive = (pred * target).sum()
    false_negative = target.sum() - true_positive
    recall = true_positive / (true_positive + false_negative + 1e-8)
    return recall.item()

def intersection_over_union(output, target, threshold=0.5):
    """
    Calculate Intersection over Union (IoU) for binary segmentation.

    Parameters:
    - output (torch.Tensor): Predicted segmentation.
    - target (torch.Tensor): Ground truth segmentation.
    - threshold (float): Threshold.

    Returns:
    - float: IoU value.
    """
    pred = (output > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = intersection / (union + 1e-8)
    return iou.item()

def dice_coefficient(output, target, threshold=0.5):
    """
    Calculate Dice Coefficient for binary segmentation.

    Parameters:
    - output (torch.Tensor): Predicted segmentation.
    - target (torch.Tensor): Ground truth segmentation.
    - threshold (float): Threshold.

    Returns:
    - float: Dice Coefficient value.
    """
    pred = (output > threshold).float()
    intersection = (pred * target).sum()
    dice = (2. * intersection) / (pred.sum() + target.sum() + 1e-8)
    return dice.item()

