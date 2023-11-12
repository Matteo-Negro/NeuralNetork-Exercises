import os
import shutil
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder

# Achitecture of the Model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 8, 5)
        self.max_pool1 = nn.MaxPool2d(4, 4)

        self.conv2 = nn.Conv2d(8, 16, 5)
        self.max_pool2 = nn.MaxPool2d(5, 5)

        self.conv3 = nn.Conv2d(16, 32, 5)
        self.max_pool3 = nn.MaxPool2d(5, 5)

        self.lin1 = nn.Linear(32, 128)
        self.lin2 = nn.Linear(128, 84)
        self.lin3 = nn.Linear(84, 9)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.max_pool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool2(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.max_pool3(x)

        x = torch.flatten(x, 1)

        x = self.lin1(x)
        x = F.relu(x)

        x = self.lin2(x)
        x = F.relu(x)

        x = self.lin3(x)

        return x
    

# Define a function to calculate accuracy
def calculate_accuracy(loader):
    correct = 0
    total = 0
    with torch.no_grad():
        running_loss = 0
        for data in loader:
            inputs, labels = data
            outputs = net(inputs.to(device))
            running_loss += torch.nn.CrossEntropyLoss()(outputs.to(device), labels.to(device)).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()
    return (100 * correct / total), running_loss


# Define the initial dataset directory with all the images
dataset_directory = 'dataset'

# Define the output directory for saving the training and testing data
output_directory = 'dataset-refactor'
train_dir = os.path.join(output_directory, 'training')
test_dir = os.path.join(output_directory, 'testing')

# Define the dump dataset directory where I will sore the dups
dump_directory = 'dataset-dumb'
dump_train_dir = os.path.join(dump_directory, 'training')
dump_test_dir = os.path.join(dump_directory, 'testing')

if not os.path.isdir(dump_directory):
    # Create subdirectories for training and testing data
    os.makedirs(train_dir)
    os.makedirs(test_dir)

    # Create the directories for training and testing data
    os.makedirs(dump_train_dir)
    os.makedirs(dump_test_dir)

    # Define the number of images per class for training and testing
    images_per_class       = 10000
    images_per_class_train = 8000
    images_per_class_test  = images_per_class - images_per_class_train

    # List of class names
    class_names = sorted(['Circle', 'Square', 'Octagon', 'Heptagon', 'Nonagon', 'Star', 'Hexagon', 'Pentagon', 'Triangle'])

    # Loop through each class
    for class_name in class_names:
        # Create the folder
        os.makedirs(f'{train_dir}/{class_name}')
        os.makedirs(f'{test_dir}/{class_name}')

        # List all image files in the dataset directory
        image_files = [f for f in os.listdir(dataset_directory) if f.endswith('.png') and class_name in f]

        # Randomly shuffle the image files
        random.shuffle(image_files)

        # Split the image files into training and testing sets
        train_images = image_files[:images_per_class_train]
        test_images = image_files[images_per_class_train:]

        # Copy images to the appropriate directories
        for image in train_images:
            src = os.path.join(dataset_directory, image)
            dst = os.path.join(f'{train_dir}/{class_name}', image)
            shutil.copy(src, dst)

        for image in test_images:
            src = os.path.join(dataset_directory, image)
            dst = os.path.join(f'{test_dir}/{class_name}', image)
            shutil.copy(src, dst)

    # Create datasets and data loaders
    transform     = transforms.Compose([transforms.Resize((200, 200)), transforms.ToTensor()])
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    test_dataset  = datasets.ImageFolder(test_dir, transform=transform)

    # Save the datasets to files
    torch.save(train_dataset, dump_train_dir + '/training.file')
    torch.save(test_dataset, dump_test_dir + '/testing.file')

# Data structures to store the loss and the accuracy
train_loss = list()
train_acc  = list()
test_loss  = list()
test_acc   = list()

# Hyper-parameters
learning_rate = 0.001
batch_size    = 100
epochs        = 10
device        = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Define data transformations
transform = transforms.Compose([transforms.ToTensor()])

# Define the same neural network architecture as in the training code
net = Net().to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

# Load the train and test files
training_files = torch.load(dump_train_dir + '/training.file')
trainloader = torch.utils.data.DataLoader(training_files, batch_size=batch_size, shuffle=True)

testing_files = torch.load(dump_test_dir + '/testing.file')
testloader = torch.utils.data.DataLoader(testing_files, batch_size=batch_size, shuffle=False)

# Train the neural network
for epoch in range(epochs):
    running_loss = 0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = net(inputs.to(device))
        loss = criterion(outputs.to(device), labels.to(device))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    acc, _ = calculate_accuracy(trainloader)
    train_acc.append(acc)
    train_loss.append(running_loss / len(trainloader))

    acc, loss = calculate_accuracy(testloader)
    test_acc.append(acc)
    test_loss.append(loss / len(testloader))
    
    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}, Train Accuracy: {train_acc[-1]}%, Test Accuracy: {test_acc[-1]}%')

print('Finished Training')
torch.save(net.state_dict(), '0502-651340543-NEGRO.ZZZ')

plt.plot(train_loss, c = 'b', label='Train Loss')
plt.plot(test_loss, c = 'r', label='Test Loss')
plt.legend()
plt.title('Train Loss & Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('loss.png',dpi=400, bbox_inches='tight', transparent=True)
plt.clf()

plt.plot(train_acc, c = 'b', label='Train Accuracy')
plt.plot(test_acc, c = 'r', label='Test Accuracy')
plt.legend()
plt.title('Train Accuracy & Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.savefig('/accuracy.png', dpi=400, bbox_inches='tight', transparent=True)
plt.clf()
