import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

input_size = 28 *28 #(MNIST images are 28 * 28 size)
num_classes = 10
batch_size = 100
num_epochs = 10
lr = 0.001

train_data = torchvision.datasets.MNIST(root='../../data',train = True,transform=transforms.ToTensor(),download=True)
test_dataset = torchvision.datasets.MNIST(root='../../data', 
                                          train=False, 
                                          transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size = batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)


model = nn.Linear (input_size,num_classes)  #(using the linear model)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = lr)

total = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1,input_size)


        # forward pass
        output = model(images)
        loss = criterion(output, labels)

        #backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('epoch[{}/{}],step [{}/{}], Loss:{:.4f}'.format(epoch+1,num_epochs,i+1,total, loss.item()))

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, input_size)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

images, labels = next(iter(test_loader))
image = images[5]
label = labels[5]

# prediction
image_flat = image.view(-1, 28*28)
output = model(image_flat)
_, predicted = torch.max(output, 1)

# show images
plt.imshow(image.squeeze(), cmap='gray')
plt.title(f'True: {label.item()}, Predicted: {predicted.item()}')
plt.show()