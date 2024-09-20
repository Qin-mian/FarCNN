import matplotlib.pyplot as plt
from torch.optim import Adam, Adagrad
from dataloader import get_data_loaders
from model import *
import torch
import numpy as np
from sklearn.metrics import f1_score

# 可调参数
SAVE = False
SEED = 111
epochs = 50
Feature = 'boundary' # change this to fit different tasks
batch_size = 64
image_size = (150, 150)
image_shape = (1, image_size[0], image_size[1])

with open(f'{Feature}.txt', 'r') as f:
    best_f1 = float(f.readline())
print(f'Before Best F1 score: {best_f1}')

# 随机种子
torch.manual_seed(SEED)
np.random.seed(SEED)

# 分类
CLASS_TYPES = ['0', '1']
N_TYPES = len(CLASS_TYPES)

# 加载数据
feature_dir = f'./{Feature}'
train_loader, test_loader = get_data_loaders(feature_dir, batch_size, image_size)

print(f'Training data loader created with batch size: {batch_size}')
print(f'Testing data loader created with batch size: {batch_size}')

# GPU训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using {device} device.')

# 初始化模型
model = CNNModel(image_shape).to(device)


# 优化器目前Adagrad最好
optimizer = Adagrad(model.parameters(), lr=0.001, lr_decay=0)
# optimizer = Adam(model.parameters(), lr=0.001)


# 损失
criterion = nn.CrossEntropyLoss()


# 早停
class EarlyStopping:
    def __init__(self, patience=10, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.best_loss = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - 1e-9:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
        else:
            self.best_loss = val_loss
            self.counter = 0

# 学习率衰减
class ReduceLROnPlateau:
    def __init__(self, factor=0.3, patience=5, verbose=False):
        self.factor = factor
        self.patience = patience
        self.verbose = verbose
        self.best_loss = None
        self.counter = 0

    def __call__(self, val_loss, optimizer):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - 1e-9:
            self.counter += 1
            if self.counter >= self.patience:
                if self.verbose:
                    print('Reducing learning rate.')
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= self.factor
                self.best_loss = val_loss
                self.counter = 0
        else:
            self.best_loss = val_loss
            self.counter = 0

def calculate_f1_score(model, dataloader):
    model.eval()
    predictions, labels = [], []
    with torch.no_grad():
        for inputs, labels_batch in dataloader:
            inputs, labels_batch = inputs.to(device), labels_batch.to(device)
            outputs = model(inputs)
            predictions.append(outputs.argmax(dim=1).cpu().numpy())
            labels.append(labels_batch.cpu().numpy())
    predictions = np.concatenate(predictions)
    labels = np.concatenate(labels)
    f1 = f1_score(labels, predictions, average='macro')
    return f1


def calculate_loss(model, dataloader, criterion):
    model.eval()
    loss = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss += criterion(outputs, labels).item()
    return loss / len(dataloader)

def calculate_accuracy(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


patience = 15
factor = 0.3
early_stopping = EarlyStopping(patience=patience, verbose=True)
reduce_lr = ReduceLROnPlateau(factor=factor, patience=patience, verbose=True)


#开始训练
for epoch in range(epochs):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss = loss
        loss.backward()
        optimizer.step()
    val_loss =  calculate_loss(model, test_loader, criterion)
    train_loss = calculate_loss(model, train_loader, criterion)
    train_acc = calculate_accuracy(model, train_loader)
    val_acc = calculate_accuracy(model, test_loader)
    print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

    f1 = calculate_f1_score(model, test_loader)
    print(f'Epoch {epoch + 1}/{epochs}, F1 Score: {f1:.4f}')
    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), f'./models/{Feature}.pth')
        print("save the best model")
        with open(f'{Feature}.txt', 'w') as f:
            f.write(f'{best_f1}')
    early_stopping(val_loss)
    reduce_lr(val_loss, optimizer)
    if early_stopping.early_stop:
        break