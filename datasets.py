import numpy as np
import random
import albumentations as A
import cv2
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

def get_images_split(root_path, ratio_train, seed=42):
    """
    Divide images into training and validation sets.

    :param root_path: Path to the root directory containing 'train_images' and 'train_masks' folders.
    :param ratio_train: Ratio of images to be included in the training set.
    :param seed: Seed for randomization.
    :return: Lists of paths for training and validation images and masks.
    """
    random.seed(seed)
    np.random.seed(seed)

    root_path = Path(root_path)
    images = sorted(root_path.glob("train_images/*"))
    masks = sorted(root_path.glob("train_masks/*"))

    assert len(images) == len(masks), f"Le nombre d'images ({len(images)}) et de masques ({len(masks)}) ne correspond pas."

    num_train = int(len(images) * ratio_train)
    indices = list(range(len(images)))
    random.shuffle(indices)
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]

    train_images = [images[i] for i in train_indices]
    train_masks = [masks[i] for i in train_indices]
    val_images = [images[i] for i in val_indices]
    val_masks = [masks[i] for i in val_indices]

    return train_images, train_masks, val_images, val_masks



def train_transforms(img_size):
    """
    Transforms/augmentations for training images and masks.

    :param img_size: Integer, for image resize.
    """
    train_image_transform = A.Compose([
            A.Resize(img_size, img_size, always_apply=True),
            A.OneOf(
                [
                    A.HorizontalFlip(p=0.8),
                    A.VerticalFlip(p=0.4),
                ],
                p=0.5,
            ),
            A.OneOf(
                [
                    A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0, p=1, border_mode=0), # scale only
                    A.ShiftScaleRotate(scale_limit=0, rotate_limit=30, shift_limit=0.1, p=1, border_mode=0), # rotate only
                    A.ShiftScaleRotate(scale_limit=0, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0), # shift only
                    A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=30, shift_limit=0.1, p=1, border_mode=0), # affine transform
                ],
                p=0.9,
            ),
            A.OneOf(
                [
                    A.Perspective(p=1),
                    A.GaussNoise(p=1),
                    A.Sharpen(p=1),
                    A.Blur(blur_limit=3, p=1),
                    A.MotionBlur(blur_limit=3, p=1),
                ],
                p=0.2,
            ),
             A.OneOf(
                [
                    A.ElasticTransform(alpha=120, sigma=150, alpha_affine=50, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, approximate=False, same_dxdy=False, p=1),
                    A.GridDistortion(num_steps=5, distort_limit=0.3, interpolation=1, border_mode=4, value=None, mask_value=None, normalized=False, always_apply=False, p=1),
                    A.Perspective(scale=(0.05, 0.1), keep_size=True, pad_mode=0, pad_val=0, mask_pad_val=0, fit_output=False, interpolation=1, always_apply=False, p=1)
                ],
                p=0.2,
            ),           
        ])
    return train_image_transform

def valid_transforms(img_size):
    """
    Transforms/augmentations for validation images and masks.

    :param img_size: Integer, for image resize.
    """
    valid_image_transform = A.Compose([
        A.Resize(img_size, img_size, always_apply=True),
    ])
    return valid_image_transform

class SegmentationDataset(Dataset):
    """
    Dataset class for image segmentation.
    """

    def __init__(self, images, masks, augmentation=None, normalize=True):
        """
        Initialize the dataset.

        :param images: List of image paths.
        :param masks: List of mask paths.
        :param augmentation: Augmentation transformations.
        :param normalize: Flag for image normalization.
        """
        self.images = images
        self.masks = masks
        self.augmentation = augmentation
        self.normalize = normalize

    def __getitem__(self, i):
        # read data
        image = cv2.imread(str(self.images[i]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self.masks[i]), 0)
        mask = mask / 255.0
        mask = np.expand_dims(mask, axis=-1)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        if self.normalize:
            image = image / 255.0

        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))

        return image, mask

    def __len__(self):
        return len(self.images)

def get_dataset(train_image_paths, train_mask_paths, valid_image_paths, valid_mask_paths, img_size):
    """
    Create training and validation datasets.

    :param train_image_paths: List of paths to training images.
    :param train_mask_paths: List of paths to training masks.
    :param valid_image_paths: List of paths to validation images.
    :param valid_mask_paths: List of paths to validation masks.
    :param img_size: Integer, for image resize.
    :return: Training and validation datasets.
    """
    train_tfms = train_transforms(img_size)
    valid_tfms = valid_transforms(img_size)

    train_dataset = SegmentationDataset(train_image_paths, train_mask_paths, train_tfms)
    valid_dataset = SegmentationDataset(valid_image_paths, valid_mask_paths, valid_tfms)

    return train_dataset, valid_dataset

def get_data_loaders(train_dataset, valid_dataset, batch_size):
    """
    Create data loaders for training and validation.

    :param train_dataset: Training dataset.
    :param valid_dataset: Validation dataset.
    :param batch_size: Batch size.
    :return: Training and validation data loaders.
    """
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=False)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, drop_last=False)

    return train_data_loader, valid_data_loader
