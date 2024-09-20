import csv
import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
import numpy as np
from torchvision import transforms

from model import CNNModel

image_shape = (1, 150, 150)
# 创建模型实例
model_boundary = CNNModel(image_shape)
model_calcification = CNNModel(image_shape)
model_direction = CNNModel(image_shape)
model_shape = CNNModel(image_shape)
# 加载模型权重
model_boundary.load_state_dict(torch.load(f'./models/boundary.pth'))
model_calcification.load_state_dict(torch.load(f'./models/calcification.pth'))
model_direction.load_state_dict(torch.load(f'./models/direction.pth'))
model_shape.load_state_dict(torch.load(f'./models/shape.pth'))
# 将模型移动到正确的设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_boundary = model_boundary.to(device)
model_calcification = model_calcification.to(device)
model_direction = model_direction.to(device)
model_shape = model_shape.to(device)


def preprocess_image(image_path):
    image = Image.open(image_path).convert('L')
    transform = transforms.Compose([
        transforms.Resize(image_shape[1:]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.456], std=[0.224])
    ])
    image = transform(image).unsqueeze(0)
    return image

def predict_image(model, image):
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        output = model(image)
        _, predicted = torch.max(output.data, 1)
    return predicted.item()

def predict_folder(folder_path):
    results = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(folder_path, filename)
            image = preprocess_image(image_path)
            result_boundary = predict_image(model_boundary, image)
            result_calcification = predict_image(model_calcification, image)
            result_direction = predict_image(model_direction, image)
            result_shape = predict_image(model_shape, image)
            results.append([result_boundary, result_calcification, result_direction, result_shape])
    return results

file_path = 'classification/classification/train/4C'
clss = '1'
predictions = predict_folder(file_path)
# 指定CSV文件的路径
csv_file_path = 'predictions.csv'

# 打开CSV文件，准备写入
with open(csv_file_path, mode='a', newline='') as file:
    writer = csv.writer(file)
    # 遍历predictions列表，写入每一行的数据
    for prediction in predictions:
        boundary, calcification, direction, shape = prediction[0:]  # 其余的是预测结果
        writer.writerow([boundary, calcification, direction, shape, clss])