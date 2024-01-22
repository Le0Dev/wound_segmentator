from matplotlib import pyplot as plt
import torch
import os

def save_training_plots(train_history, save_folder):
    """
    Save training and validation loss plots, as well as plots for each metric.

    Parameters:
    - train_history (list): List of dictionaries containing training history.
    - save_folder (str): Folder to save the plots.
    """
    # Create the folder if it doesn't exist
    os.makedirs(save_folder, exist_ok=True)

    # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    train_loss = [entry['train_loss'] for entry in train_history]
    valid_loss = [entry['valid_loss'] for entry in train_history]
    plt.plot(range(1, len(train_history) + 1), train_loss, label='Train Loss')
    plt.plot(range(1, len(train_history) + 1), valid_loss, label='Valid Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_folder, 'loss_plot.png'))
    plt.close()

    # Plot each metric for training and validation
    metrics_names = list(train_history[0]['train_metrics'].keys())
    for metric_name in metrics_names:
        plt.figure(figsize=(10, 6))
        train_metric = [entry['train_metrics'][metric_name] for entry in train_history]
        valid_metric = [entry['valid_metrics'][metric_name] for entry in train_history]
        plt.plot(range(1, len(train_history) + 1), train_metric, label=f'Train {metric_name}')
        plt.plot(range(1, len(train_history) + 1), valid_metric, label=f'Valid {metric_name}')
        plt.title(f'Training and Validation {metric_name}')
        plt.xlabel('Epoch')
        plt.ylabel(metric_name)
        plt.legend()
        plt.savefig(os.path.join(save_folder, f'{metric_name}_plot.png'))
        plt.close()

    print(f"Plots saved in {save_folder}")

def save_best_models(model, optimizer, save_folder, epoch, val_epoch_loss, val_iou):
    """
    Save the best models based on validation loss and IoU.

    Parameters:
    - model: PyTorch model.
    - optimizer: PyTorch optimizer.
    - save_folder (str): Folder to save the best models.
    - epoch (int): Current epoch.
    - val_epoch_loss (float): Validation loss for the epoch.
    - val_iou (float): Jaccard index for the epoch.
    """
    # Create the folder if it doesn't exist
    os.makedirs(save_folder, exist_ok=True)

    # Save the model based on validation loss
    if val_epoch_loss < save_best_models.best_loss:
        # Delete the previous model if it exists
        if save_best_models.best_loss_epoch > 0:
            old_path = os.path.join(save_folder, f'best_model_loss_epoch_{save_best_models.best_loss_epoch}.pth')
            os.remove(old_path)

        save_best_models.best_loss = val_epoch_loss
        save_best_models.best_loss_epoch = epoch
        save_path_loss = os.path.join(save_folder, f'best_model_loss_epoch_{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metric': val_epoch_loss
        }, save_path_loss)

        print(f"Saved Best Loss checkpoint: {val_epoch_loss:.3f}")

    # Save the model based on IoU
    if val_iou > save_best_models.best_iou:
        # Delete the previous model if it exists
        if save_best_models.best_iou_epoch > 0:
            old_path = os.path.join(save_folder, f'best_model_iou_epoch_{save_best_models.best_iou_epoch}.pth')
            os.remove(old_path)

        save_best_models.best_iou = val_iou
        save_best_models.best_iou_epoch = epoch
        save_path_iou = os.path.join(save_folder, f'best_model_iou_epoch_{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metric': val_iou
        }, save_path_iou)

        print(f"Saved Best IoU checkpoint: {val_iou:.3f}")

# Initialize the best metrics
save_best_models.best_loss = float('inf')
save_best_models.best_iou = 0.0
save_best_models.best_loss_epoch = 0
save_best_models.best_iou_epoch = 0

