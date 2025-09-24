import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import torch.utils.data as data

import torchvision.transforms as transforms
import torchvision.datasets as datasets

import numpy as np
import random
import os
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from model import MyResNet18

from loss.FocalLoss import FocalLoss

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(model, optimizer, criterion, train_loader, device, epoch= 0, log_interval=50):
    model.train()
    total_acc, total_count = 0, 0
    losses = []

    train_bar = tqdm(train_loader, unit = 'batch', desc = f"Epoch {epoch:3d}")
    for inputs, labels in train_bar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        predictions = model(inputs)

        loss = criterion(predictions, labels)
        losses.append(loss.item())

        #backward
        loss.backward()
        optimizer.step()

        total_acc += (predictions.argmax(1) == labels).sum().item()
        total_count += labels.size(0)

        #update train_bar
        train_bar.set_postfix(
            loss= f"{loss.item():.4f}",
            acc = f"{(total_acc/total_count)*100:.2f}%"
        )
    epoch_acc = total_acc / total_count
    epoch_loss = np.mean(losses)
    # Display final result of epoch
    print(f"Epoch: {epoch}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc*100:.2f}%")
    return epoch_loss, epoch_acc

#Evaluation function
def evaluate(model, criterion, valid_dataloader):
    model.eval()
    total_acc, total_count = 0, 0
    losses = []
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(valid_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            predictions = model(inputs)
            loss = criterion(predictions, labels)
            losses.append(loss.item())
            total_acc += (predictions.argmax(1) == labels).sum().item()
            total_count += labels.size(0)
    epoch_acc = total_acc/total_count
    epoch_loss = sum(losses)/len(losses)
    return epoch_acc, epoch_loss

def main():

    seed = 68
    set_seed(seed)

    IMG_SIZE = 224
    BATCH_SIZE = 64

    main_folder = os.getcwd()
    data_dir = os.path.join(main_folder,'cassavaleafdata')
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'validation')
    test_dir = os.path.join(data_dir, 'test')

    initial_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ]
    )

    #Create Dataset object
    # Using ImageFolder for training
    dataset_for_mean_std = datasets.ImageFolder(train_dir, transform=initial_transform)

    # Create dataloader to load in batch

    loader_for_mean_std = DataLoader(dataset_for_mean_std, batch_size=BATCH_SIZE, shuffle=False)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_pixels = 0

    for images, _ in tqdm(loader_for_mean_std, desc="Tinh mean/std"):
        images = images.permute(0, 2, 3, 1).contiguous().view(-1, 3)
        mean += images.sum(dim = 0)
        total_pixels += images.shape[0]

    mean /= total_pixels
    for images, _ in tqdm(loader_for_mean_std, desc="Tinh std"):
        images = images.permute(0, 2, 3, 1).contiguous().view(-1, 3)
        std += ((images - mean)**2).sum(dim=0)
        total_pixels += images.shape[0]

    std /= total_pixels
    std = torch.sqrt(std)

    print(f"Mean: {mean}")
    print(f"Std: {std}")

    

    print(mean, std)

    img_size = 224
    train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)

    ])
    tesr_val_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform=tesr_val_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=tesr_val_transforms)

    labels = train_dataset.classes
    print(labels)
    label_to_idx = train_dataset.class_to_idx
    print(label_to_idx)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    num_classes = len(labels)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MyResNet18(num_classes).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Using Focal loss 
    # number each label
    cbb = 466
    cbsd = 1443
    cgm = 773
    cmd = 2658
    healthy = 316
    total_sample = cbb + cbsd + cgm + cmd + healthy
    alpha_tensor = torch.Tensor([1/ (cbb/total_sample), 1/ (cbsd/total_sample), 1/ (cgm/total_sample), 1/ (cmd/total_sample), 1/ (healthy/total_sample)])
    alpha_tensor = alpha_tensor/alpha_tensor.sum()
    alpha_tensor = alpha_tensor.to(device)
    criterion = FocalLoss(alpha=alpha_tensor, gamma=2.0, reduction='mean')

    num_epochs  = 10
    save_model = './models'
    os.makedirs(save_model, exist_ok=True)

    train_accs = []
    train_losses = []
    eval_accs = []
    eval_losses = []
    best_loss_eval = 10000
    tolerance = 0
    epoch = 1

    while tolerance < 5:
        train_loss, train_acc = train(model, optimizer, criterion,
                                    train_loader, device, epoch)

        train_accs.append(train_acc)
        train_losses.append(train_loss)

        #Evaluation
        eval_acc, eval_loss = evaluate(model, criterion, val_loader)
        eval_accs.append(eval_acc)
        eval_losses.append(eval_loss)

        #Save best model
        if eval_loss < best_loss_eval:
            best_loss_eval = eval_loss
            torch.save(model.state_dict(), os.path.join(save_model, 'best_modelCrossEntropy.pth'))
            tolerance = 1
        elif eval_loss > best_loss_eval:
            tolerance += 1
        epoch += 1

    #print loss, acc and epoch
    print ("-" * 59)
    #Load th best model
    model.load_state_dict(torch.load(os.path.join(save_model, 'best_modelCrossEntropy.pth')))
    model.eval()
    epoch += 1

    test_acc, test_loss = evaluate(model, criterion, test_loader)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc*100:.2f}%")

if __name__ == '__main__':
    main()