import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from functools import partial
from data.data_utils import load_nii_nn
import pandas as pd

def check_normal(root, filtered_df):
    for subject_id in filtered_df['subject_id']:
        if subject_id in root:
            return False
    return True

def find_nii_directories(base_dir, csv_path, modality="FLAIR"):
    """
    Recursively find all directories containing .nii.gz files with specific criteria.

    Args:
        base_dir (str): The base directory to search.

    Returns:
        list: List of directories containing at least one .nii.gz file with the modality.
    """
    nii_directories = list()
    anomal_directories = list()
    df = pd.read_csv(csv_path, sep=',', quotechar='"') 
    filtered_df = df[df['PXPERIPH'] == 2]

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".nii.gz") and modality in file and "cleaned" in file:
                if check_normal(root, filtered_df):
                    nii_directories.append(os.path.join(root, file))
                    break
                else:
                    anomal_directories.append(os.path.join(root, file))
                    break
    return nii_directories, anomal_directories

class ADNIDataset(Dataset):
    def __init__(self, directories, transform=None, labels=None, config=None) -> None:
        self.directories = directories        
        self.config = config

    def __len__(self):
        return len(self.directories)
    
    def __getitem__(self, index):
        volume = load_nii_nn(self.directories[index], )
        return volume

def get_dataloaders(train_base_dir, csv_path, modality, batch_size=4, transform=None,
                    validation_split=0.1, test_split=0.1, seed=42, config=None, inf=False):
    """
    Prepare and return DataLoaders for training, validation, and testing.

    Args:
        train_base_dir (str): Base directory containing training NIfTI directories.
        batch_size (int, optional): Batch size for DataLoaders. Default is 4.
        transform (callable, optional): Transformations to apply to the data.
        validation_split (float, optional): Fraction of the dataset to use for validation.
        test_split (float, optional): Fraction of the dataset to use for testing.
        seed (int, optional): Random seed for reproducibility.
        config (Namespace, optional): Configuration for loading parameters.

    Returns:
        tuple: (train_loader, validation_loader, test_loader)
    """
    if transform is None:
        transform = transforms.Compose([
            transforms.Normalize((0.5,), (0.5,)),
        ])
    torch.manual_seed(seed)

    print("Data load....")

    train_directories, anomal_directories = find_nii_directories(train_base_dir, csv_path,modality)
    train_directories = train_directories[:10]        
    # if inf is True:
    #     train_imgs = np.concatenate(load_images(anomal_directories, config))
    #     validation_dataset = TrainDataset(train_imgs)
    #     validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    #     return None, validation_loader, None


    #train_imgs = np.concatenate(load_images(train_directories, config))
    train_dataset =ADNIDataset(directories=train_directories, transform=transform, labels=None, config=config)
    total_size = len(train_dataset)
    validation_size = int(total_size * validation_split)
    test_size = int(total_size * test_split)    
    train_size = total_size - validation_size - test_size

    train_dataset, validation_dataset, test_dataset = random_split(train_dataset, [train_size, validation_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Len train_loader: {len(train_loader)}")
    print(f"Len val_loader: {len(validation_loader)}")
    print(f"Len test_loader: {len(test_loader)}")

    return train_loader, validation_loader, test_loader