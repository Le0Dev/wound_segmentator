"""
                                 __                                        __        __            
 _      ______  __  ______  ____/ /  ________  ____ _____ ___  ___  ____  / /_____ _/ /_____  _____
| | /| / / __ \/ / / / __ \/ __  /  / ___/ _ \/ __ `/ __ `__ \/ _ \/ __ \/ __/ __ `/ __/ __ \/ ___/
| |/ |/ / /_/ / /_/ / / / / /_/ /  (__  )  __/ /_/ / / / / / /  __/ / / / /_/ /_/ / /_/ /_/ / /    
|__/|__/\____/\__,_/_/ /_/\__,_/  /____/\___/\__, /_/ /_/ /_/\___/_/ /_/\__/\__,_/\__/\____/_/     
                                            /____/                                                 

This code was written as part of a personal project. The idea is to create a semi-automated image annotation pipeline. 
In this context, the first part of this project consists of carrying out a binary semantic segmentation of the wounds of an image. 
As it stands, all of the code has not yet been completely written, there are still various optimizations and pieces of code to write 
but the latter should allow you to train a functional model!

Command-line example: 'python train.py --data_path "./data/" --learning_rate 0.0001 --batch_size 8 --epochs 100 --img_size 256'

Not yet implemented in command-line arguments:
    - models.py contains other networks (U_Net, R2U_Net, R2AttU_Net), you can use these architectures by calling them in this script.
    - You are also invited to modify the scheduler as you wish directly in the script
    
Not implemented: reproducible experiment config file, inference.py (check the notebook inference.ipynb)
Not implemented optimizations: TTA, reproducible experiment config file
"""

import torch
import argparse
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datasets import get_images_split, get_dataset
from models import AttU_Net, U_Net, R2U_Net, R2AttU_Net
from losses import DiceBCELoss
from engine import train, validate
from utils import save_training_plots, save_best_models

def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Train an AttU-Net model for segmentation.")
    parser.add_argument("--data_path", type=str, default="./data/", help="Path to the data directory.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training and validation.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--img_size", type=int, default=256, help="Image size for preprocessing.")
    
    return parser.parse_args()

def main(args):
    """
    Main function for training an AttU-Net model on a segmentation task.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        None
    """
    DATA_PATH = args.data_path
    
    # PARAMETERS
    LEARNING_RATE = args.learning_rate
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    IMG_SIZE = args.img_size
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load and split images
    train_images, train_masks, valid_images, valid_masks = get_images_split(
        root_path=DATA_PATH, ratio_train=0.8
    )

    # Create datasets
    train_dataset, valid_dataset = get_dataset(
        train_images, 
        train_masks,
        valid_images,
        valid_masks,
        img_size=IMG_SIZE
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model, optimizer, loss function, and learning rate scheduler
    model = AttU_Net(3, 1).to(device) # change to use other models
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = DiceBCELoss()
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
    
    # Print model parameters information
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(device)
    print(f"{total_params:,} total parameters {total_trainable_params:,} training parameters.")
    print(f'EPOCHS: {EPOCHS}, LR: {LEARNING_RATE}, BS: {BATCH_SIZE}')
    
    # Training loop
    train_history = []
    for epoch in range(EPOCHS):
        print(f"EPOCH: {epoch + 1}")

        # Train & Validation
        train_epoch_loss, train_metrics = train(model, train_loader, device, optimizer, criterion)
        valid_epoch_loss, valid_metrics = validate(model, val_loader, device, criterion)
        save_best_models(model, optimizer, 'models', epoch, valid_epoch_loss, valid_metrics['iou'])
    
        # Print metrics
        print(
            f"Train Loss: {train_epoch_loss:.3f}, Acc: {train_metrics['accuracy']:.3f}, Precision: {train_metrics['precision']:.3f}, "
            f"Recall: {train_metrics['recall']:.3f}, mIOU: {train_metrics['iou']:.3f}, Dice: {train_metrics['dice']:.3f}",
            f"\nValid Loss: {valid_epoch_loss:.3f}, Acc: {valid_metrics['accuracy']:.3f}, Precision: {valid_metrics['precision']:.3f}, "
            f"Recall: {valid_metrics['recall']:.3f}, mIOU: {valid_metrics['iou']:.3f}, Dice: {valid_metrics['dice']:.3f}"
        )
        
        # Adjust learning rate based on validation loss
        scheduler.step(valid_epoch_loss)
        
        # Save epoch information for later analysis
        epoch_info = {
            'epoch': epoch + 1,
            'train_loss': train_epoch_loss,
            'train_metrics': train_metrics,
            'valid_loss': valid_epoch_loss,
            'valid_metrics': valid_metrics
        }

        train_history.append(epoch_info)
    # Save training plots
    save_training_plots(train_history, './plots/')

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
    