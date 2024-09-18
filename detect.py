import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
import numpy as np
from model import CNNModel

Feature = 'calcification'

image_shape = (1, 150, 150)
# 创建模型实例
model = CNNModel(image_shape)
# 加载模型权重
model.load_state_dict(torch.load(f'./models/{Feature}.pth'))
# 将模型移动到正确的设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 创建GUI窗口
window = tk.Tk()
window.title("Image Classifier")

# 创建图像显示区域
image_label = tk.Label(window)
image_label.pack()

# 创建预测结果显示区域
result_label = tk.Label(window)
result_label.pack()

# 定义选择图像的函数
def select_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        # 加载图像
        image = Image.open(file_path)
        image = image.resize((150, 150))
        image = image.convert('L')  # 转换为灰度图像
        image = np.array(image)
        image = np.expand_dims(image, axis=0)  # 增加一个通道维度
        image = image / 255.0
        image = torch.tensor(image, dtype=torch.float32)
        image = image.unsqueeze(0)  # 增加批次维度
        image = image.to(device)

        # 进行预测
        with torch.no_grad():
            output = model(image)
        _, predicted = torch.max(output, 1)

        # 显示预测结果
        result_label.config(text=f'Predicted: {predicted.item()}')

        # 显示图像
        image = image[0].cpu().numpy().squeeze()  # 移除批次和通道维度
        image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        image_label.config(image=image)
        image_label.image = image

# 创建选择图像的按钮
select_button = tk.Button(window, text="Select Image", command=select_image)
select_button.pack()

# 运行GUI窗口
window.mainloop()
