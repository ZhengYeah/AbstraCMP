import os
import torch
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets


BATCH_SIZE = 64

training_data = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)
train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE)

testing_data = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)
test_dataloader = DataLoader(testing_data, batch_size=BATCH_SIZE)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 80),
            nn.ReLU(),
            nn.Linear(80, 80),
            nn.ReLU(),
            nn.Linear(80, 80),
            nn.ReLU(),
            nn.Linear(80, 80),
            nn.ReLU(),
            nn.Linear(80, 80),
            nn.ReLU(),
            nn.Linear(80, 80),
            nn.ReLU(),
            nn.Linear(80, 80),
            nn.ReLU(),
            nn.Linear(80, 80),
            nn.ReLU(),

            nn.Linear(80, 10)
        )
    def forward(self, x):
        x = self.linear_relu_stack(x)
        return x


model = NeuralNetwork()


def train(model, dataloader):
    size = len(dataloader.dataset)
    learning_rate = 1e-3
    epochs = 20
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        for batch, (x, y) in enumerate(dataloader):
            x = x.reshape(-1, 784)
            # Compute prediction and loss
            pred = model(x)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(x)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    print("Done!")

    if not os.path.isdir('checkpoint'):
        os.mkdir('./checkpoint')
    torch.save(model.state_dict(), './checkpoint/mnist_net.pth')


def generate_properties(model_path, testloader):
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    # print('Ground Truth:', ' '.join('%s' % labels[j].item() for j in range(len(labels))))
    
    # net = NeuralNetwork()
    # net.load_state_dict(torch.load(model_path))

    # outputs = net(images)
    # _, predicted = torch.max(outputs, 1)
    # print('Predicted:', ' '.join('%s' % predicted[j].item() for j in range(len(predicted))))

    images = images.reshape(-1, 784)
    for i in range(BATCH_SIZE):
        with open("./mnist_properties/mnist_property_" + str(i) + ".txt", "w") as img_file:
            for j in range(784):
                img_file.write("%.8f\n" % images[i][j].item())
            for k in range(10):
                if k == labels[i].item():
                    continue
                property_list = [0 for _ in range(11)]
                property_list[labels[i]] = -1
                property_list[k] = 1
                img_file.write(str(property_list)[1:-1].replace(',', ''))
                img_file.write('\n')


def test_accuracy(model_path, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, lables = data

            net = NeuralNetwork()
            net.load_state_dict(torch.load(model_path))

            images = images.view(-1, 784)
            images = images.squeeze()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += lables.size(0)
            correct += (predicted == lables).sum().item()    # item() for getting the element of a tensor
    print("Accuracy: %d %%" % (correct * 100 / total))


if __name__ == "__main__":
    train(model, train_dataloader)
    test_accuracy("checkpoint/mnist_net.pth", test_dataloader)

    # generate_properties("checkpoint/mnist_net.pth", test_dataloader)
