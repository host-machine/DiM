import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, inSize, outSize):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inSize, outSize, kernel_size=3, padding=1),
            nn.BatchNorm2d(outSize),
            nn.ReLU(inplace=True),
            nn.Conv2d(outSize, outSize, kernel_size=3, padding=1),
            nn.BatchNorm2d(outSize),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, inSize, outSize):
        super().__init__()
        self.conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(inSize, outSize))

    def forward(self, x):
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, inSize, outSize):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(inSize, outSize, 1)

    def forward(self, x):
        return self.conv(x)


class Up(nn.Module):
    def __init__(self, inSize, outSize):
        super().__init__()

        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv = DoubleConv(inSize, outSize)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)



class UNet(nn.Module):
    def __init__(self, nChannel, nClass):
        super(UNet, self).__init__()
        self.inc = DoubleConv(nChannel, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, nClass)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)








from PIL import Image
import os
import numpy as np
from torch.utils.data import Dataset

class ImgData(Dataset):
    def __init__(self, data_path):
        self.path = data_path
        self.imgForder = os.path.join(data_path, "image")
        self.imgName = os.listdir(self.imgForder)

    # 加载图像
    def loadImg(self, path):
        img = np.array(Image.open(path))
        return img.reshape(1, *img.shape)

    # 根据index读取图片
    def __getitem__(self, index):
        #pImg = os.path.join(self.path, f"image\{index}.png")
        #pLabel = os.path.join(self.path, f"label\{index}.png")
        pImg = os.path.join(self.path, f"image\{self.imgName[index]}")
        pLabel = os.path.join(self.path, f"label\{self.imgName[index]}")

        image = self.loadImg(pImg)
        label = self.loadImg(pLabel)
        # 数据标签归一化
        if label.max() > 1:
            label = label / 255
        # 随机翻转图像，增加训练样本
        flipCode = np.random.randint(3)
        if flipCode!=0:
            image = np.flip(image, flipCode).copy()
            label = np.flip(label, flipCode).copy()
        return image, label

    def __len__(self):
        # 返回训练集大小
        return len(os.listdir(self.imgForder))




from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn

def train(net, device, path, epochs=40, bSize=1, lr=0.00001):
    igmData = ImgData(path)
    train_loader = DataLoader(igmData, bSize, shuffle=True)
    # 优化算法
    optimizer = optim.RMSprop(net.parameters(),
            lr=lr, weight_decay=1e-8, momentum=0.9)

    criterion = nn.BCEWithLogitsLoss()      # 损失函数
    bestLoss = float('inf')                # 最佳loss，初始化为无穷大

    # 训练epochs次
    for epoch in range(epochs):
        net.train()     # 训练模式
        for image, label in train_loader:
            optimizer.zero_grad()
            # 将数据拷贝到device中
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)

            pred = net(image)   # 使用网络参数，输出预测结果
            loss = criterion(pred, label)   # 计算损失
            # 保存loss最小的网络参数
            if loss < bestLoss:
                bestLoss = loss
                torch.save(net.state_dict(), 'best_model.pth')

            loss.backward() # 更新参数
            optimizer.step()

        print(epoch, 'Loss/train', loss.item())


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = UNet(1, 1)
net.to(device=device)
path = "D:\work\Data\DiMTAIC\ImageData"
epoch = 3900
batch = 1
learn_rate = 0.000001
train(net, device, path, epoch, batch, learn_rate)














#predict
def predictOne(net, device, pRead, pSave):
    img = Image.open(pRead)
    img = np.array(img)
    img = img.reshape(1, 1, *img.shape)

    img = torch.from_numpy(img)
    img = img.to(device=device, dtype=torch.float32)

    pred = net(img)  # 预测
    pred[pred >= 0.5] = 255
    pred[pred < 0.5] = 0

    pred = np.array(pred.data.cpu()[0])[0]
    img = Image.fromarray(pred.astype(np.uint8))
    img.save(pSave)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = UNet(1, 1)
net.to(device=device)
net.load_state_dict(torch.load('best_model.pth', map_location=device))

net.eval()  # 测试模式
fs = os.listdir(r'D:\work\Data\DiMTAIC\exam\test')
for f in fs:
    pRead = os.path.join(r"D:\work\Data\DiMTAIC\exam\test", f)
    pSave = os.path.join(r"D:\work\Data\DiMTAIC\exam\predict", f)
    predictOne(net, device, pRead, pSave)




