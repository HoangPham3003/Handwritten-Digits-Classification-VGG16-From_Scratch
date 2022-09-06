import os
import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm
import torch.nn.functional as F

from vgg16 import VGG16
from datasets import HandwrittenDigitsDatasetLoader


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_classes = 10
num_epochs = 50
batch_size = 64
learning_rate = 0.005
weight_decay = 0.005


model = VGG16(num_classes=num_classes)

# model = torchvision.models.vgg16(pretrained=True)
# model.classifier[-1] = nn.Linear(in_features=4096, out_features=10)
model = model.to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# dataset loader
train = False
dataloader = None

if train:
    dataloader = HandwrittenDigitsDatasetLoader(batch_size=batch_size, train=True)
    train_loader, valid_loader = dataloader.load_data()

    # training phase
    print("Start training...")

    iterations = len(train_loader)

    loss_opt = 1e9
    for epoch in range(num_epochs):
        loss = None
        for i, (images, labels) in tqdm(enumerate(train_loader)):
            images = images.to(device)
            labels = labels.to(device)

            # forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' .format(epoch+1, num_epochs, i+1, iterations, loss.item()))
        if loss < loss_opt:
            loss_opt = loss
            torch.save(model.state_dict(), './best.pth')

        # validating phase
        with torch.no_grad():
            # turn off model.train() -> turn on model.eval() -> turn off model.eval() -> and then auto turn on model.train() again
            correct = 0
            total = 0
            for images, labels in valid_loader:
                """
                    image: tensor
                    label: tensor
                """
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                outputs = F.softmax(outputs, dim=1)
                value_softmax, index_softmax = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (index_softmax == labels).sum().item()
                del images, labels, outputs
            print('Accuracy of the network on the validation images: {} %'.format(100 * correct / total))
else:
    model = VGG16(num_classes=num_classes)
    # model = torchvision.models.vgg16()
    # model.classifier[-1] = nn.Linear(in_features=4096, out_features=10)
    model.load_state_dict(torch.load('./best.pth'))
    model = model.to(device) 

    dataloader = HandwrittenDigitsDatasetLoader(batch_size=1, train=False)
    test_loader = dataloader.load_data()

    # testing phase
    with torch.no_grad():
        # turn off model.train() -> turn on model.eval() -> turn off model.eval() -> and then auto turn on model.train() again
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            outputs = F.softmax(outputs, dim=1)
            value_softmax, index_softmax = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (index_softmax == labels).sum().item()
            del images, labels, outputs
        print('Accuracy of the network on the testing images: {} %'.format(100 * correct / total))
