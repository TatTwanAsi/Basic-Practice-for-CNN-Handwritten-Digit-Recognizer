import torchvision
import torch
from tqdm import tqdm
import time

def train(model, device):

    model = model.to(device)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,))
    ])
    

    trainset = torchvision.datasets.MNIST(root = "./", train = True, download = True, transform = transform)
    trainset_loader = torch.utils.data.DataLoader(trainset, shuffle = True, batch_size = 2048, num_workers = 4, pin_memory = True)

    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.005)
    loss_list = []

    for epoch in range(10):
        print(f"epoch{epoch+1}")
        loss_sum = 0
        # progress_bar = tqdm(trainset_loader, desc = "training progress")
        start_time = time.time()
        for i, batch in enumerate(trainset_loader, 1):
            features, labels = batch
            features, labels = features.to(device), labels.to(device)

            yhats = model(features)
            loss = criterion(yhats, labels)
            loss_sum += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # avg_loss = loss_sum / i
            # progress_bar.set_postfix({'Loss': f'{avg_loss:.3f}'})

        end_time = time.time()
        average_loss = loss_sum/len(trainset_loader)
        loss_list.append(average_loss)
        print(f"time cost: {end_time - start_time:.2f} s")
        print(f"average loss: {average_loss}")
        print("---------------------------------------------")
    
    
    return model, loss_list
