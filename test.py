import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score
from dataloader import get_data_loaders
from model import CNNModel


Feature = 'calcification'  # 根据你的实际特征名称进行替换
feature_dir = f'./{Feature}'
batch_size = 64
image_size = (150,150)
image_shape = (1, image_size[0], image_size[1])
N_TYPES = 2
_, test_loader = get_data_loaders(feature_dir, batch_size, image_size)

# 创建模型实例
model = CNNModel(image_shape)
# 加载模型权重
model.load_state_dict(torch.load(f'./models/{Feature}.pth'))
# 将模型移动到正确的设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 进行预测
test_predictions = []
test_labels = []

model.eval()  # 设置模型为评估模式
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        test_predictions.extend(predicted.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())

# 计算混淆矩阵
confusion_mtx = confusion_matrix(test_labels, test_predictions)
# 假设你有一个二分类问题，类别标签为0和1
class_indices_train_list = ['negative', 'positive']  # 根据你的实际类别进行替换

# 计算精度
accuracy = accuracy_score(test_labels, test_predictions)
print(f"Accuracy: {accuracy:.4f}")

# 计算召回率
recall = recall_score(test_labels, test_predictions, average='macro')
print(f"Recall: {recall:.4f}")

# 计算F1得分
f1 = f1_score(test_labels, test_predictions, average='macro')
print(f"F1 Score: {f1:.4f}")

# 绘制混淆矩阵
fig, ax = plt.subplots(figsize=(8, 8))
cax = ax.matshow(confusion_mtx, cmap=plt.cm.Blues, alpha=0.7)
fig.colorbar(cax)

# 添加文本标签和标题
ax.set_title(f'{Feature} Confusion Matrix')
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_xticklabels(class_indices_train_list, ha='center', rotation=45)
ax.set_yticklabels(class_indices_train_list)

# 在混淆矩阵中添加数据标签
for i in range(N_TYPES):
    for j in range(N_TYPES):
        ax.text(j, i, format(confusion_mtx[i, j], 'd'),
                ha="center", va="center", color="black")

plt.savefig(f'./{Feature}_confusion_matrix.png')
plt.show()