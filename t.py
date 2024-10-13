import torch
from pathlib import Path
import requests
import pickle
import gzip
from matplotlib import pyplot
import numpy as np
from setuptools.namespaces import flatten
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image


#换到GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

#
# # 定义数据集的路径
# PATH = Path(r"C:\Users\50597\Desktop\mnist.pkl.gz")
#
# # 使用gzip打开文件
# with gzip.open(str(PATH), 'rb') as f:
#     # 加载数据集
#     ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")
#
# #拿数据并转为tensor
# x_train, y_train, x_valid, y_valid = map(torch.tensor, (x_train, y_train, x_valid, y_valid))
# n, c = x_train.shape
# x_train, x_train.shape, y_train.min(), y_train.max()
#
# #定义关键参数
# loss_func = F.cross_entropy
# bs = 64
# xb = x_train[0:bs]  # a mini-batch from x
# yb = y_train[0:bs]
# weights = torch.randn([784, 10], dtype = torch.float,  requires_grad = True)
# bias = torch.zeros(10, requires_grad=True)
#
#
# #定义网络
class Mnist_NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(784, 256)
        self.hidden2 = nn.Linear(256, 128)
        self.hidden3 = nn.Linear(128, 256)
        self.out = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = self.out(x)
        return x

#数据储存对象
# train_ds = TensorDataset(x_train, y_train)
# valid_ds = TensorDataset(x_valid, y_valid)

#以储存对象做参数创建数据迭代对象
def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )

#
def get_model():
    model = Mnist_NN()
    model.to(device)
    return model, optim.Adam(model.parameters(), lr=0.001)

def loss_batch(model, loss_func, xb, yb, opt=None):
    xb, yb = xb.to(device), yb.to(device)
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)


#定义训练函数
def fit(steps, model, loss_func, opt, train_dl, valid_dl):
    for step in range(steps):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        print('当前step:'+str(step), '验证集损失：'+str(val_loss))
#
#
#
#
# train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
# model, opt = get_model()
# fit(15, model, loss_func, opt, train_dl, valid_dl)
#
# #保存模型参数
# torch.save(model.state_dict(), 'model_state_dict.pth')

# 加载模型参数
model = Mnist_NN()  # 重新创建模型实例
model.load_state_dict(torch.load('model_state_dict.pth'))
model.eval()
model.to(device)  # 确保模型在正确的设备上

#加载新图像
image = Image.open(r"C:\Users\50597\Desktop\OIP-C (2).jpg")  # 替换为新图像的路径

# 定义预处理转换
transform = transforms.Compose([
    transforms.Grayscale(),  # 转为灰度图像
    transforms.Resize((28, 28)),  # 调整图像大小
    transforms.ToTensor(),  # 转为张量
    transforms.Normalize((0.5,), (0.5,))  # 归一化
])


# 应用预处理
image = transform(image).unsqueeze(0)
image = image.view(-1)
image = image.to(device)
#前向传播

# 用torch.no_grad()进行预测，这样不会计算梯度
with torch.no_grad():
    output = model(image)

print(output)
print(torch.argmax(output, 0))
# print(f'Predicted label: {predicted.item()}')













