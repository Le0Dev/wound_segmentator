import torch
from tqdm import tqdm
from metrics import accuracy, precision, recall, intersection_over_union, dice_coefficient

class EvalMetrics:
    """
    Class for managing metric evaluation during training.
    """
    def __init__(self):
        """
        Initializes an instance of the EvalMetrics class.
        """
        self.reset()

    def reset(self):
        """
        Resets metrics at the beginning of each epoch.
        """
        self.metrics = {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'iou': 0.0,
            'dice': 0.0
        }
        self.counter = 0

    def update(self, output, target):
        """
        Updates metrics with new values.

        Parameters:
            - output (torch.Tensor): Predicted values.
            - target (torch.Tensor): Target values (ground truth).
        """
        self.metrics['accuracy'] += accuracy(output, target)
        self.metrics['precision'] += precision(output, target)
        self.metrics['recall'] += recall(output, target)
        self.metrics['iou'] += intersection_over_union(output, target)
        self.metrics['dice'] += dice_coefficient(output, target)
        self.counter += 1

    def compute(self):
        """
        Computes average metric values for the entire training/evaluation period.

        Returns:
            dict: Average metric values.
        """
        for key in self.metrics.keys():
            self.metrics[key] /= self.counter
        return self.metrics

def process_data(data, device):
    """
    Prepares data for training/evaluation by moving it to the specified device.

    Parameters:
        - data (tuple): Data to process.
        - device (torch.device): Computing device (CPU or GPU).

    Returns:
        tuple: Processed data ready for training/evaluation.
    """
    return data[0].float().to(device), data[1].float().to(device)

def train(model, train_dataloader, device, optimizer, criterion):
    """
    Performs a training epoch for the model.

    Parameters:
        - model (torch.nn.Module): Model to train.
        - train_dataloader (torch.utils.data.DataLoader): DataLoader for training data.
        - device (torch.device): Computing device (CPU or GPU).
        - optimizer (torch.optim.Optimizer): Optimizer to use.
        - criterion (torch.nn.Module): Loss function to minimize.

    Returns:
        tuple: Training loss and overall training metrics.
    """
    model.train()
    num_batches = len(train_dataloader)
    prog_bar = tqdm(train_dataloader, total=num_batches, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}', desc='Training')
    eval_metrics_train = EvalMetrics()

    for data in prog_bar:
        
        data, target = process_data(data, device)
        optimizer.zero_grad()
        outputs = torch.sigmoid(model(data))
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            eval_metrics_train.update(outputs, target)
            
    train_loss = loss.item()
    overall_metrics_train = eval_metrics_train.compute()
    return train_loss, overall_metrics_train

def validate(model, valid_dataloader, device, criterion):
    """
    Performs a validation epoch for the model.

    Parameters:
        - model (torch.nn.Module): Model to evaluate.
        - valid_dataloader (torch.utils.data.DataLoader): DataLoader for validation data.
        - device (torch.device): Computing device (CPU or GPU).
        - criterion (torch.nn.Module): Loss function to use for evaluation.

    Returns:
        tuple: Validation loss and overall validation metrics.
    """
    model.eval()
    num_batches = len(valid_dataloader)
    eval_metrics_val = EvalMetrics()

    with torch.no_grad():
        prog_bar = tqdm(valid_dataloader, total=num_batches, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}', desc='Evaluation')
        for data in prog_bar:
            data, target = process_data(data, device)
            outputs = torch.sigmoid(model(data))
            loss = criterion(outputs, target)
            eval_metrics_val.update(outputs, target)

    valid_loss = loss.item()
    overall_metrics_val = eval_metrics_val.compute()
    return valid_loss, overall_metrics_val

