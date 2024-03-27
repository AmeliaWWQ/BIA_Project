from skimage import io
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchmetrics


class GlauDataset(Dataset):
    def __init__(self, is_train, transform=None):
        self.is_train = is_train
        self.transform = transform

        csv_file = './Kmeans_2000.csv'
        df = pd.read_csv(csv_file)
        ids = df['Eye ID'].tolist()
        labels = df['Final Label'].tolist()
        ids = [_.split('/')[-1] for _ in ids]

        pos_cases = []
        neg_cases = []
        for id, lab in zip(ids, labels):
            if lab == 1:
                pos_cases.append((id, lab))
            elif lab == 0:
                neg_cases.append((id, lab))

        num_pos_tr = int(len(pos_cases) * 0.8)
        num_neg_tr = int(len(neg_cases) * 0.8)
        self.train_data = pos_cases[:num_pos_tr] + neg_cases[:num_neg_tr]
        self.val_data = pos_cases[num_pos_tr:] + neg_cases[num_neg_tr:]

    def __len__(self):
        if self.is_train:
            return len(self.train_data)
        else:
            return len(self.val_data)

    def __getitem__(self, idx):
        if self.is_train:
            img_path, lab = self.train_data[idx]
            img = io.imread('/fast/yangz16/glaucoma/train_images/' + img_path)
            img = img.astype(np.float32)
            img = img.transpose(2, 0, 1)  # todo
            img = torch.from_numpy(img)
            img = self.transform(img)
        else:
            img_path, lab = self.val_data[idx]
            img = io.imread('/fast/yangz16/glaucoma/train_images/' + img_path)
            img = img.astype(np.float32)
            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img)

        lab = torch.tensor(lab, dtype=torch.float32)
        return img, lab


class ResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        # model = torchvision.models.resnet50(pretrained=True)
        # self.model = torchvision.models.resnet50(weights=None)
        self.model = torch.hub.load("pytorch/vision", "resnet50", weights="IMAGENET1K_V2")
        self.model.fc = nn.Linear(2048, 1)

    def forward(self, x):
        return self.model(x).squeeze()


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set hyperparameters
num_epochs = 10
batch_size = 24
learning_rate = 0.001

# Initialize transformations for data augmentation
transform = transforms.Compose([
    transforms.Resize(512, antialias=True),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=45),
    # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    # transforms.CenterCrop(224),
    # transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the ImageNet Object Localization Challenge dataset
# train_dataset = torchvision.datasets.ImageFolder(
#     root='/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train',
#     transform=transform
# )
train_dataset = GlauDataset(is_train=True, transform=transform)
val_dataset = GlauDataset(is_train=False)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=10)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=10)

# Load the ResNet50 model
# model = torchvision.models.resnet50(pretrained=True)
# model = torchvision.models.resnet50(weights=None)
# model = torch.hub.load("pytorch/vision", "resnet50", weights="IMAGENET1K_V2")
model = ResNet50()

# Parallelize training across multiple GPUs
# model = torch.nn.DataParallel(model)

# Set the model to run on the device
model = model.to(device)

# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=num_epochs, power=1.0)

# Metric
metric = torchmetrics.classification.BinaryAUROC()
metric.to(device)

# Train the model...
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        # Move input and label tensors to the device
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero out the optimizer
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

    scheduler.step()
    for param_group in optimizer.param_groups:
        current_lr = param_group['lr']
    # Print the loss for every epoch
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, LR: {current_lr:.6f}')

    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            metric.update(outputs, labels)

    auroc = metric.compute()
    print(f'Epoch {epoch+1}/{num_epochs}, AUROC: {auroc:.4f}')

print(f'Finished Training, Loss: {loss.item():.4f}')