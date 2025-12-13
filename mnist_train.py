import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# ====================== 1. 配置参数 ======================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 优先使用GPU
BATCH_SIZE = 64        # 批次大小
EPOCHS = 5             # 训练轮数
LEARNING_RATE = 0.001  # 学习率

# ====================== 2. 数据预处理与加载 ======================
# 数据变换：转为张量 + 归一化（MNIST像素值0-255，归一化到0-1）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST数据集的均值和标准差
])

# 加载MNIST数据集（自动下载）
train_dataset = datasets.MNIST(
    root='./data',  # 数据保存路径
    train=True,     # 训练集
    download=True,  # 自动下载
    transform=transform
)

test_dataset = datasets.MNIST(
    root='./data',
    train=False,    # 测试集
    download=True,
    transform=transform
)

# 数据加载器（批量加载+打乱）
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 可视化单个样本
def show_sample():
    # 取第一个样本
    image, label = train_dataset[0]
    # 转换为numpy格式（CHW -> HWC）
    image = image.numpy().squeeze()
    plt.imshow(image, cmap='gray')
    plt.title(f"Label: {label}")
    plt.axis('off')
    plt.show()

show_sample()  # 运行查看样本

# ====================== 3. 构建神经网络模型 ======================
class HandwritingCNN(nn.Module):
    def __init__(self):
        super(HandwritingCNN, self).__init__()
        # 卷积层：提取图像特征
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 输入通道1（灰度图），输出32通道
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 池化：缩小特征图尺寸
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        # 全连接层：分类
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),  # 卷积后特征图尺寸：7x7（28/2/2）
            nn.ReLU(),
            nn.Dropout(0.5),  # 防止过拟合
            nn.Linear(128, 10)  # 输出10类（0-9）
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # 展平特征图
        x = self.fc_layers(x)
        return x

# 初始化模型、损失函数、优化器
model = HandwritingCNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()  # 交叉熵损失（分类任务）
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ====================== 4. 模型训练 ======================
def train(model, loader, criterion, optimizer, epoch):
    model.train()  # 训练模式
    total_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        
        # 前向传播
        output = model(data)
        loss = criterion(output, target)
        total_loss += loss.item()
        
        # 反向传播+优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 计算准确率
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        
        # 打印进度
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch} [{batch_idx * len(data)}/{len(loader.dataset)}] '
                  f'Loss: {loss.item():.6f}')
    
    # 计算本轮平均损失和准确率
    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / len(loader.dataset)
    print(f'Train Epoch {epoch}: Average loss: {avg_loss:.4f}, Accuracy: {correct}/{len(loader.dataset)} ({accuracy:.2f}%)')

# ====================== 5. 模型测试 ======================
def test(model, loader, criterion):
    model.eval()  # 评估模式（关闭Dropout）
    test_loss = 0
    correct = 0
    with torch.no_grad():  # 禁用梯度计算（加速）
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    avg_loss = test_loss / len(loader)
    accuracy = 100. * correct / len(loader.dataset)
    print(f'Test set: Average loss: {avg_loss:.4f}, Accuracy: {correct}/{len(loader.dataset)} ({accuracy:.2f}%)\n')
    return accuracy

# ====================== 6. 执行训练和测试 ======================
if __name__ == '__main__':
    best_accuracy = 0
    for epoch in range(1, EPOCHS + 1):
        train(model, train_loader, criterion, optimizer, epoch)
        test_acc = test(model, test_loader, criterion)
        
        # 保存最优模型
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(),r'.\model\MNIST\best_handwriting_model.pth')
    
    #print(f'Best test accuracy: {best_accuracy:.2f}%')

# ====================== 7. 自定义图片预测（可选） ======================
from PIL import Image

def predict_number_image(image_path):
    # 加载并预处理图片（转为灰度、调整尺寸为28x28、归一化）
    image = Image.open(image_path).convert('L')  # 转为灰度图
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    image = transform(image).unsqueeze(0).to(DEVICE)  # 增加batch维度
    
    # 预测
    model.load_state_dict(torch.load(r'.\model\MNIST\best_handwriting_model.pth'))
    model.eval()
    with torch.no_grad():
        output = model(image)
        pred = output.argmax(dim=1).item()
    
    print(f'Predicted digit: {pred}')
    # 可视化预测的图片
    img = Image.open(image_path).convert('L')
    plt.imshow(img, cmap='gray')
    plt.title(f'Predicted: {pred}')
    plt.axis('off')
    plt.show()
    return pred

# 调用示例（替换为你的手写数字图片路径，建议白底黑字、28x28尺寸）
#predict_custom_image(r'.\test\1.png')