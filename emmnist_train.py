import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# ====================== 1. 配置参数 ======================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 0.001
NUM_CLASSES = 62  # 62类（0-9+A-Z+a-z）

# ====================== 2. 数据预处理 ======================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.transpose(1, 2)),  # 修正EMNIST图片方向
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载EMNIST数据集
train_dataset = datasets.EMNIST(
    root='./data',
    split='byclass',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.EMNIST(
    root='./data',
    split='byclass',
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 标签映射函数
def label_to_char(label):
    if 0 <= label <= 9:
        return str(label)
    elif 10 <= label <= 35:
        return chr(label - 10 + ord('A'))
    elif 36 <= label <= 61:
        return chr(label - 36 + ord('a'))
    else:
        return "未知"

# 可视化样本
def show_emnist_sample():
    image, label = train_dataset[100]
    image = image.numpy().squeeze()
    plt.imshow(image, cmap='gray')
    plt.title(f"Label: {label} → {label_to_char(label)}")
    plt.axis('off')
    plt.show()

show_emnist_sample()

# ====================== 3. 模型定义 ======================
class HandwritingCNN(nn.Module):
    def __init__(self, num_classes):
        super(HandwritingCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

# ====================== 4. 初始化模型/优化器/调度器（关键：正确实例化scheduler） ======================
model = HandwritingCNN(NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ✅ 正确实例化学习率调度器（必须加括号，传入优化器和参数）
scheduler = optim.lr_scheduler.StepLR(
    optimizer, 
    step_size=5,  # 每5轮学习率减半
    gamma=0.5     # 学习率衰减系数
)

# ====================== 5. 训练/测试函数 ======================
def train(model, loader, criterion, optimizer, epoch):
    model.train()
    total_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        output = model(data)
        loss = criterion(output, target)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

        if batch_idx % 200 == 0:
            print(f'Epoch {epoch} [{batch_idx * len(data)}/{len(loader.dataset)}] '
                  f'Loss: {loss.item():.6f}')

    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / len(loader.dataset)
    print(f'Train Epoch {epoch}: Average loss: {avg_loss:.4f}, Accuracy: {correct}/{len(loader.dataset)} ({accuracy:.2f}%)')

def test(model, loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
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

# ====================== 6. 执行训练（关键：scheduler.step() 放在循环内） ======================
if __name__ == '__main__':
    best_accuracy = 0
    for epoch in range(1, EPOCHS + 1):
        train(model, train_loader, criterion, optimizer, epoch)
        test_acc = test(model, test_loader, criterion)
        
        # ✅ 每轮训练后更新学习率（确保scheduler是实例，且在循环内调用）
        scheduler.step()

        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), r'.\model\MNIST\best_emnist_model.pth')
    
    print(f'Best test accuracy: {best_accuracy:.2f}%')

# ====================== 7. 自定义图片预测 ======================
from PIL import Image

def predict_custom_char(image_path):
    image = Image.open(image_path).convert('L')
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        #transforms.Lambda(lambda x: x.transpose(1, 2).flip(1)),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    image = transform(image).unsqueeze(0).to(DEVICE)

    model.load_state_dict(torch.load(r'.\model\MNIST\best_emnist_model.pth'))
    model.eval()
    with torch.no_grad():
        output = model(image)
        pred_label = output.argmax(dim=1).item()
        pred_char = label_to_char(pred_label)

    print(f'Predicted label: {pred_label} → {pred_char}')
    img = Image.open(image_path).convert('L')
    plt.imshow(img, cmap='gray')
    plt.title(f'Predicted: {pred_char}')
    plt.axis('off')
    plt.show()

predict_custom_char('write_test_image/A.png')