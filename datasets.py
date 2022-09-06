import os
import numpy as np
import torch
import torchvision
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler 


class HandwrittenDigitsDataset(Dataset):
    def __init__(self, transform=None, train=True):
        self.train = train
        self.dataset = None
        if train:
            self.dataset = torchvision.datasets.MNIST(root='./Data', train=True)
        else:
            self.dataset = torchvision.datasets.MNIST(root='./Data', train=False)
        self.transform = transform

    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index] # image: PIL image; label: int
        image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        image = np.array(image)
        return image, label


class HandwrittenDigitsDatasetLoader:
    def __init__(self, batch_size=8,
                       random_seed=42,
                       valid_size=0.2,
                       shuffle=True,
                       train=True):
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.valid_size = valid_size
        self.shuffle = shuffle
        self.train = train
        
        
    def load_data(self):
        # Create transform
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]

        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
            transforms.RandomAffine(degrees=40, translate=None, scale=(1, 2), shear=15, resample=False, fillcolor=0),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        # eval
        if not self.train:
            test_dataset = HandwrittenDigitsDataset(transfrom=test_transform, train=False)
            test_loader = DataLoader(
                dataset=test_dataset, batch_size=self.batch_size
            )
            return test_loader

        # train
        # Create dataset
        train_dataset = HandwrittenDigitsDataset(transform=train_transform, train=True)
        valid_dataset = HandwrittenDigitsDataset(transform=train_transform, train=True)

        # Train-Valid dataset split
        num_train = train_dataset.__len__()
        indices = list(range(num_train))
        split = int(np.floor(self.valid_size * num_train))

        if self.shuffle:
            np.random.seed(self.random_seed)
            np.random.seed(indices)

        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        
        # Create data loader
        train_loader = DataLoader(
            dataset=train_dataset, batch_size=self.batch_size, sampler=train_sampler
        )

        valid_loader = DataLoader(
            dataset=valid_dataset, batch_size=self.batch_size, sampler=valid_sampler
        )

        return (train_loader, valid_loader)
