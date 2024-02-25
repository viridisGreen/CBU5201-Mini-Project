# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from CustomDataset import Genki4kDataset
from PIL import Image
from tqdm import tqdm

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device: ', device)

# Hyperparameters
learning_rate = 0.001
batch_size = 64
num_epochs = 16
lr_dacay_epochs = 4
criterion = nn.CrossEntropyLoss()

# Load data
data_dir = 'genki4k/files'
label_file = 'genki4k/labels.txt'
transform = transforms.Compose([
    transforms.Lambda(lambda x: Image.fromarray(x)),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = Genki4kDataset(data_dir=data_dir, label_file=label_file, transform=transform)
# train_set, test_set = torch.utils.data.random_split(dataset, [3200, 800])
train_set, vali_set, test_set = torch.utils.data.random_split(dataset, [3000, 500, 500])

train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
vali_loader = DataLoader(dataset=vali_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

print('Data loaded successfully')

# Checking accuracy
def check_accuracy(loader, model):
    if loader == train_loader:
        print("Checking accutacy on training data: ")
    elif loader == vali_loader:
        print("Checking accutacy on validation data: ")
    else:
        print("Checking accuracy on test data: ")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y, _ in tqdm(loader):
            x = x.to(device)
            y = y.to(device)

            scores = model(x)
            loss = criterion(scores, y)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100}')





#--------------------------------------------------------------------------------------#




# Define model
# model = torchvision.models.resnet18(pretrained = True, num_classes=1000)
# model.fc = nn.Linear(model.fc.in_features, 2)
model = torchvision.models.mobilenet_v2(pretrained = True)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Train
train_losses = []
vali_losses = []
for epoch in range(num_epochs):
    if (epoch+1) % lr_dacay_epochs == 0:
        learning_rate = learning_rate * 0.1

    # On training data
    losses = []
    for i, (data, label, _) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training')):
        data = data.to(device)
        label = label.to(device)

        scores = model(data)
        loss = criterion(scores, label)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if (i+1)%5 == 0:
        #     print(f'epoch [{(epoch+1):2d} / {num_epochs}]: iteration [{(i+1):2d} / {len(train_loader)}] is done')
    train_losses.append(sum(losses))

    # On validation data
    losses = []
    for i, (data, label, _) in enumerate(vali_loader):
        with torch.no_grad():
            data = data.to(device)
            label = label.to(device)

            scores = model(data)
            loss = criterion(scores, label)
            losses.append(loss.item())
    vali_losses.append(sum(losses))

    print(f'Epoch {epoch+1}/{num_epochs} - Loss: {sum(losses) / len(losses)}\n')


# Check accuracy
check_accuracy(train_loader, model)
check_accuracy(test_loader, model)


# Plotting the training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(vali_losses, label='Validation Loss')
plt.title('Training vs. Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.savefig('/home/whs/smile_loss.png')















