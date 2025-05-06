import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

input_size = 1
output_size = 1
num_epochs = 400 #(best epoch number)
lr = 0.01

# train dataset
x_train = np.array([[2.0], [2.643], [3.286], [3.929], [4.571], 
                    [5.214], [5.857], [6.5], [7.143], [7.786], 
                    [8.429], [9.071], [9.714], [10.357], [11.0]], dtype=np.float32)

y_train = np.array([[2.40], [1.98], [2.95], [2.32], [3.97],
                    [3.31], [4.45], [3.62], [4.89], [4.24], 
                    [5.91], [5.10], [6.38], [5.77], [7.23]], dtype=np.float32)



model = nn.Linear(input_size,output_size)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(num_epochs):

    inputs = torch.from_numpy(x_train)
    targets = torch.from_numpy(y_train)  # real data

    outputs = model(inputs)   # prediction
    loss = criterion(outputs,targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 5 == 0:
        print ('Epoch [{}/{}], Loss : {: .4f}'.format(epoch+1, num_epochs, loss.item()))

# plot the graph 

predicted = model(torch.from_numpy(x_train)).detach().numpy()
plt.plot(x_train, y_train, 'ro', label='Original data')
plt.plot(x_train, predicted, label='Fitted line')
plt.legend()
plt.show()


