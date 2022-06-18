import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

batch_size = 1

#pixel values of range [0, 1] on dataset

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #Tensors of normalized range [-1, 1]

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,download=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Function for save and load model
def save_checkpoint(state, filename = 'CNN.pth.tar'):
    print('Saving checkpoints')
    torch.save(state, filename)

def load_checkpoint(checkpoint):
    print('Loading checkpoint')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

# Create the network class
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.layer1 = nn.Linear(16 * 5 * 5, 120)
        self.layer2 = nn.Linear(120, 84)
        self.layer3 = nn.Linear(84, 20)
        self.out = nn.Linear(20, 10)

    def forward(self, x):
        # Convolutions and pooling
        
        # x.shape  -> [1, 3, 32, 32]
        # Convolutional Layers
        x = self.pool(F.relu(self.conv1(x)))  # -> n, 6, 14, 14
        x = self.pool(F.relu(self.conv2(x)))  # -> n, 16, 5, 5
        x = x.view(-1, 16 * 5 * 5)            # -> n, 400
        
        # NN
        x = F.relu(self.layer1(x))            # -> n, 120
        x = F.leaky_relu(self.layer2(x))      # -> n, 84
        x = F.relu(self.layer3(x))
        x = self.out(x)                       # -> n, 10
        return x
    
    def fit(self, num_epochs):
        # Training
        self.losses = []
        n_total_steps = len(train_loader)
        for epoch in range(num_epochs):
            if epoch+1 == num_epochs:
                checkpoint = {'state_dict' : model.state_dict(), 'optimizer' : optimizer.state_dict()}
                save_checkpoint(checkpoint)

            for i, (images, labels) in enumerate(train_loader):
                # origin shape: [4, 3, 32, 32] = 4, 3, 1024
                # input_layer: 3 input channels, 6 output channels, 5 kernel size
                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(images.float())
                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if (i+1) % 2000 == 0:
                    print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
                    self.losses.append(loss.item())

    def evaluate(self):
        with torch.no_grad():
            # check accuracy
            n_correct = 0
            n_samples = 0
            n_class_correct = [0 for i in range(10)]
            n_class_samples = [0 for i in range(10)]

            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                # max returns (value ,index)
                _, predicted = torch.max(outputs, 1)
                n_samples += labels.size(0)
                n_correct += (predicted == labels).sum().item()
                
                for i in range(batch_size):
                    label = labels[i]
                    pred = predicted[i]
                    if (label == pred):
                        n_class_correct[label] += 1
                    n_class_samples[label] += 1

            acc = 100.0 * n_correct / n_samples
            print(f'Accuracy of the network: {acc} %')

            for i in range(10):
                acc = 100.0 * n_class_correct[i] / n_class_samples[i]
                print(f'Accuracy of {classes[i]}: {acc} %')
        fig = px.line(x = [i for i in range(len(self.losses))], y = self.losses)
        fig.show()
    
    def looking_at_camera(self ):
        ######## Connect to camera
        import cv2
        import torch
        import numpy as np

        cap = cv2.VideoCapture(0)
        while True:
            # get image
            _, image1 = cap.read()

            # make compatible
            image_ = cv2.resize(image1, (32,32)) # resize

            '''The model is applying a normalization and a standardization (channel1, channel3, channel3)'''
            image_ = transform(image_) # apply normalization
            image = torch.tensor(image_).reshape(3,32,32) # reshape
            image = torch.tensor(image.unsqueeze(0))      # put inside a container

            #transmit to Cnn
            image.to(device) 
            prediction = model.forward(image.float())
            prediction = classes[torch.argmax(prediction)]
            print(prediction)

            cv2.putText(
                    img = image1,
                    text = prediction,
                    org = (200, 200),
                    fontFace = cv2.FONT_HERSHEY_DUPLEX,
                    fontScale = 3.0,
                    color = (125, 246, 55),
                    thickness = 3
                    )
            cv2.imshow('Image', image1)
            cv2.waitKey(1)

# Create model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ConvNet().float().to(device)

# Hyper-parameters 
learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Main logic
load_model = False

if load_model:
    load_checkpoint(torch.load('CNN.pth.tar'))
else:
    model.fit(10)
    model.evaluate()

model.looking_at_camera()
