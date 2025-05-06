import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


batch_size = 100
lr = 0.001
input_size = 28 *28
hidden_size = 500
num_classes = 10
num_epochs = 10


# minst dataset

train_data = torchvision.datasets.MNIST(root='../data', train=True,transform=transforms.ToTensor(),download=True)
test_data = torchvision.datasets.MNIST(root='../data', train=False, transform=transforms.ToTensor())

# dataloader

train_loader = torch.utils.data.DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data,batch_size=batch_size,shuffle=False)

# feedforward neural network

class NeuralNet(nn.Module):
    def __init__(self,input_size,hidden_size,num_classes):
        super(NeuralNet,self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size,num_classes)

    def forward(self,x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)

        return out
    
model = NeuralNet(input_size,hidden_size,num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = lr)

# train the model
total_step = len(train_data)

for epoch in range(num_epochs):
    for i, (images,labels) in enumerate (train_loader):

        images = images.reshape(-1,28*28).to(device)
        labels = labels.to(device)


        # forward
        output = model(images)
        loss = criterion(output,labels)

        # backward and optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step[{}/{}], Loss:{:.4f}'.format(epoch+1, num_epochs,i+1, total_step, loss.item()))

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))









