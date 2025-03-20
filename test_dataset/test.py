import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader

# GPUが利用可能ならGPUを使う
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# データ前処理（学習時と同じ設定にする）
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Debugモードのフラグ（Trueなら全画像について詳細出力）
debug_mode = True

if debug_mode:
    # ImageFolderを継承して画像ファイルのパスも返すクラスを定義
    class DebugImageFolder(datasets.ImageFolder):
        def __getitem__(self, index):
            sample, target = super().__getitem__(index)
            # サンプルのパスはself.samplesに格納されている (path, label) のタプル
            path, _ = self.samples[index]
            return sample, target, path

    # デバッグ用データセット（valフォルダ内の画像）
    debug_dataset = DebugImageFolder(root="data/val", transform=val_transform)
    debug_loader = DataLoader(debug_dataset, batch_size=1, shuffle=False, num_workers=0)
else:
    # 通常の評価用データセット
    val_dataset = datasets.ImageFolder(root="data/val", transform=val_transform)
    debug_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# torchvisionで提供されるResNeXtのモデル構造（学習時と同じ設定）
model = models.resnext50_32x4d(pretrained=False)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10)
model = model.to(device)

# 保存済みのモデルパラメータをロード
model.load_state_dict(torch.load("best_resnext.pth", map_location=device))
model.eval()

# クラス名のリスト（ImageFolderの場合、自動的にソートされる）
if debug_mode:
    classes = debug_dataset.classes
else:
    classes = val_dataset.classes

# デバッグモードの場合は、各画像のファイル名、正解ラベル、予測ラベルを出力
if debug_mode:
    print("Debug Mode: 全画像の推論結果を出力します。")
    with torch.no_grad():
        for inputs, labels, paths in debug_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            image_path = paths[0]  # バッチサイズ1なのでリストの最初の要素
            true_label = classes[labels.item()]
            pred_label = classes[preds.item()]
            print(f"Image: {image_path}, True Label: {true_label}, Predicted Label: {pred_label}")
else:
    # 通常の評価処理（精度計算など）
    running_corrects = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, labels in debug_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            running_corrects += torch.sum(preds == labels.data)
            total_samples += labels.size(0)
    
    accuracy = running_corrects.double() / total_samples
    print(f"Validation Accuracy: {accuracy:.4f}")
