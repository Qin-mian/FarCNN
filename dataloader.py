import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

class CustomDataset(ImageFolder):
    def __init__(self, directory, transform=None):
        super(CustomDataset, self).__init__(root=directory, transform=transform)

def get_data_loaders(data_dir, batch_size, image_size, shuffle=True, random_state=0):
    # image transformations
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = CustomDataset(os.path.join(data_dir, 'train'), transform=train_transform)
    test_dataset = CustomDataset(os.path.join(data_dir, 'test'), transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

if __name__ == "__main__":
    train_dir = 'path_to_train_data'
    batch_size = 32
    image_size = (150, 150)

    train_loader, test_loader = get_data_loaders(train_dir, batch_size, image_size)

    print(f'Training data loader created with batch size: {batch_size}')
    print(f'Testing data loader created with batch size: {batch_size}')