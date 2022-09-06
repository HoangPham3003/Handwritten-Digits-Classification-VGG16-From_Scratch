import os
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import torch.nn.functional as F

from vgg16 import VGG16
from datasets import HandwrittenDigitsDataset


class HandwrittenDigitsPredicter:
    def __init__(self, image_path=None, check_point='./best.pth'):
        self.image_path = image_path
        self.check_point = check_point
        self.device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.get_model()
        
        
    def get_model(self):
        model = VGG16(num_classes=10)
        model.load_state_dict(torch.load('./best.pth', map_location=torch.device('cpu')))
        model = model.to(self.device)
        model.eval()
        return model

    
    def predict(self):
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]

        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        image = cv2.imread(self.image_path)
        image = Image.fromarray(image)
        image = image.convert('RGB')
        image = test_transform(image)
        image = image.unsqueeze(0)
        image = image.to(self.device)

        output = self.model(image)
        output = F.softmax(output, dim=1)
        value_hat, label_hat = torch.max(output.data, dim=1)
        label_hat = np.array(label_hat)[0]
        return label_hat
