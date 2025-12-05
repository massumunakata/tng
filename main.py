import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

# =========================
# データ準備
# =========================
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# データセット読み込み
train_dataset = datasets.ImageFolder(
    "/home/nisimuramasato/ドキュメント/GitHub/tngchina/coffee_dataset",
    transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

print("Classes:", train_dataset.classes)  # ['defect', 'normal']

# =========================
# モデル定義
# =========================
class CoffeeNet(nn.Module):
    def __init__(self):
        super(CoffeeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.fc1 = nn.Linear(self._get_conv_output(), 128)
        self.fc2 = nn.Linear(128, 2)

    def _get_conv_output(self):
        # ダミー入力で畳み込み後のサイズを計算
        x = torch.zeros(1, 3, 128, 128)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        return x.numel()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# =========================
# 学習
# =========================
model = CoffeeNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 5
for epoch in range(epochs):
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# =========================
# 保存
# =========================
torch.save(model.state_dict(),
           "/home/nisimuramasato/ドキュメント/GitHub/tngchina/models/coffee_model.pth")
print("モデルを保存しました！")

# =========================
# 推論（判定）
# =========================
# 判定したい画像を指定
img_path = "/home/nisimuramasato/ドキュメント/GitHub/tngchina/coffeebeanoriginaldata/Deteksi Jenis Kopi/test/184.jpg"
img = Image.open(img_path)
img = transform(img).unsqueeze(0)

model.eval()
with torch.no_grad():
    output = model(img)
    pred = torch.argmax(output, dim=1)

classes = train_dataset.classes  # ['defect', 'normal']
print("判定結果:", classes[pred.item()])
