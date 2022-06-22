import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F


def cifar_loaders(batch_size): 
    train = datasets.CIFAR10('./data', train=True, download=False, transform=transforms.Compose([transforms.ToTensor()]))
    test = datasets.CIFAR10('./data', train=False, download=False, transform=transforms.Compose([transforms.ToTensor()]))

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader, test_loader


batch_size = 64
test_batch_size = 1000

train_loader, _ = cifar_loaders(batch_size)
_, test_loader = cifar_loaders(test_batch_size)


class LayerFC_Net(nn.Module):
    def __init__(self):
        super(LayerFC_Net, self).__init__()
        self.linear1 = torch.nn.Linear(3072, 200)
        self.linear2 = torch.nn.Linear(200, 200)
        self.linear3 = torch.nn.Linear(200, 10)

        
    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear2(x))

        x = self.linear3(x)
        return x 
      

model = LayerFC_Net()
#print(model)
num_epochs = 50


def train(model, train_loader):
    criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    for epoch in range(num_epochs):
        avg_loss_epoch = 0
        batch_loss = 0
        total_batches = 0

        for i, (images, labels) in enumerate(train_loader):
            # Reshape images to (batch_size, input_size)
            images = images.reshape(-1, 32*32*3)
            #print(images.shape)
            outputs = model(images)
            loss = criterion(outputs, labels)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()   

            total_batches += 1     
            batch_loss += loss.item()

        avg_loss_epoch = batch_loss/total_batches
        print('Epoch [{}/{}], Averge Loss:for epoch[{}, {:.4f}]'.format(epoch+1, num_epochs, epoch+1, avg_loss_epoch))

    if not os.path.isdir('checkpoint'):
        os.mkdir('./checkpoint')
    torch.save(model.state_dict(), './checkpoint/cifar_net.pth')


def test(model_path, test_loader):
    correct = 0
    total = 0

    model = LayerFC_Net()
    model.load_state_dict(torch.load(model_path))

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.reshape(-1, 3072)
            #print(labels)
            outputs_test = model(images)
            _, predicted = torch.max(outputs_test.data, 1)
            #print(predicted)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))


def generate_properties(model_path, train_loader):
    model = LayerFC_Net()
    model.load_state_dict(torch.load(model_path))
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    images = images.reshape(-1, 3072)

    outputs_test = model(images)
    _, predicted = torch.max(outputs_test.data, 1)

    i, tmp = 0, 0
    while i < 100 and tmp < 1000:
        if predicted[tmp] == labels[tmp]:
            with open("./cifar_properties/cifar_property_" + str(i) + ".txt", "w") as img_file:
                for j in range(3072):
                    img_file.write("%.8f\n" % images[tmp][j].item())
                for k in range(10):
                    if k == labels[tmp].item():
                        continue
                    property_list = [0 for _ in range(11)]
                    property_list[labels[tmp]] = -1
                    property_list[k] = 1
                    img_file.write(str(property_list)[1:-1].replace(',', ''))
                    img_file.write('\n')
            i += 1
        tmp += 1
    print(tmp)


if __name__ == '__main__':
    # train(model, train_loader)
    # test('checkpoint/cifar_net.pth', test_loader)

    generate_properties('checkpoint/cifar_net.pth', test_loader)