import torchvision
import torch

def test(model, device):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,))
    ])

    testset = torchvision.datasets.MNIST(root = "./", train = False, download = True, transform = transform)
    testset_loader = torch.utils.data.DataLoader(testset, shuffle = False, batch_size = 1000)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in testset_loader:
            features, labels = features.to(device), labels.to(device)

            outputs = model(features)
            predicted = torch.argmax(outputs, dim=1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = 100 * correct / total
    print(f"accuracy rate: {accuracy:.2f}%")
    print(f"correct prediction: {correct}/{total}")
