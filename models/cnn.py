import torch

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 3, padding = 1)
        self.conv2 = torch.nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1)
        self.pool = torch.nn.MaxPool2d(kernel_size = 2)
        self.ReLU = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(64*14*14, 1024)
        self.fc2 = torch.nn.Linear(1024, 128)
        self.fc3 = torch.nn.Linear(128, 10)
        self.dropout = torch.nn.Dropout(p = 0.25)
        self.softmax = torch.nn.LogSoftmax(dim = 1)
    
    def forward(self, x):
        x = self.ReLU(self.conv1(x))
        x = self.ReLU(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64*14*14)
        x = self.dropout(self.ReLU(self.fc1(x)))
        x = self.ReLU(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x