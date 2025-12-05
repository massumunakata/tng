import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import os

# モデル定義（学習時と同じ構造）
class CoffeeNet(nn.Module):
    def __init__(self):
        super(CoffeeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.fc1 = nn.Linear(32*62*62, 128)  # flatten後のサイズに合わせる
        self.fc2 = nn.Linear(128, 2)          # defect / normal

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# データ変換（学習時と同じ）
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 学習済みモデルをロード
model = CoffeeNet()
model.load_state_dict(torch.load(
    "/home/nisimuramasato/ドキュメント/GitHub/tngchina/models/coffee_model.pth",
    map_location=torch.device('cpu')
))
model.eval()

# クラス名（学習時の順序に依存）
classes = ['defect', 'normal']

# 判定対象フォルダ
img_dir = "/home/nisimuramasato/ドキュメント/GitHub/tngchina/coffeebeanoriginaldata/Deteksi Jenis Kopi/test"

# 1〜30.jpg を順番に判定
for i in range(1, 200):
    img_path = os.path.join(img_dir, f"{i}.jpg")
    if not os.path.exists(img_path):
        print(f"{i}.jpg は存在しません")
        continue

    img = Image.open(img_path)
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        pred = torch.argmax(output, dim=1)

    print(f"{i}.jpg → 判定結果: {classes[pred.item()]}")
