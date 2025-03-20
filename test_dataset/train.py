import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # TensorBoard用

# GPUが利用可能ならGPUを使う
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TensorBoard用のライターを初期化
writer = SummaryWriter(log_dir="runs/experiment")

# データ拡張・前処理の定義
data_transforms = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # ImageNetで学習されたモデルの正規化パラメータを利用
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]),
    "val": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
}

# ImageFolderを利用してデータセットを作成
train_dataset = datasets.ImageFolder(root="data/train", transform=data_transforms["train"])
val_dataset = datasets.ImageFolder(root="data/val", transform=data_transforms["val"])

# DataLoaderの設定
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# torchvisionの事前学習済みResNeXtモデルを読み込み
model = models.resnext50_32x4d(pretrained=True)

# 最終層の置換：元の1000クラス分類用の全結合層を2クラス用に再定義
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10)
model = model.to(device)

# 損失関数と最適化手法の定義
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 30  # エポック数（必要に応じて調整）

# 学習ループ
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    
    # 訓練フェーズ
    model.train()
    running_loss = 0.0
    running_corrects = 0
    
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)
    
    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_corrects.double() / len(train_dataset)
    print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
    
    # TensorBoardに訓練結果を記録
    writer.add_scalar('Loss/train', epoch_loss, epoch)
    writer.add_scalar('Accuracy/train', epoch_acc, epoch)
    
    # 検証フェーズ
    model.eval()
    val_loss = 0.0
    val_corrects = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            val_corrects += torch.sum(preds == labels.data)
    
    val_loss = val_loss / len(val_dataset)
    val_acc = val_corrects.double() / len(val_dataset)
    print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
    
    # TensorBoardに検証結果を記録
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Accuracy/val', val_acc, epoch)

print("Training complete")
torch.save(model.state_dict(), 'best_resnext.pth')
writer.close()
