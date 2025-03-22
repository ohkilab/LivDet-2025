import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse

from datasetloader_v2 import SiameseFingerprintDatasetV2
from model import SiameseNetwork

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="images", help="画像ディレクトリのパス")
    parser.add_argument("--epochs", type=int, default=10, help="エポック数")
    parser.add_argument("--batch_size", type=int, default=16, help="バッチサイズ")
    parser.add_argument("--lr", type=float, default=1e-4, help="学習率")
    parser.add_argument("--save_path", type=str, default="siamese_model.pth", help="学習済みモデルの保存先")
    args = parser.parse_args()

    # 画像前処理：リサイズとテンソル変換
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = SiameseFingerprintDatasetV2(root_dir=args.root_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    model_net = SiameseNetwork().to(device)

    optimizer = optim.Adam(model_net.parameters(), lr=args.lr)
    # CosineEmbeddingLoss を利用（ターゲットは正例なら 1、負例なら -1）
    criterion = nn.CosineEmbeddingLoss()

    model_net.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        for img1, img2, labels in dataloader:
            img1 = img1.to(device)
            img2 = img2.to(device)
            # データセットのラベルは 0（同一）/1（異なる）なので、1 - 2*label により 1 or -1 に変換
            target = 1 - 2 * labels.to(device)
            target = target.view(-1)  # CosineEmbeddingLoss は (N,) を要求

            optimizer.zero_grad()
            feat1, feat2 = model_net(img1, img2)
            loss = criterion(feat1, feat2, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * img1.size(0)
        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {epoch_loss:.4f}")

    torch.save(model_net.state_dict(), args.save_path)
    print(f"Model saved to {args.save_path}")

if __name__ == "__main__":
    main()
