import torch
from torchvision import transforms
from PIL import Image
import argparse
import os
import pickle

from model import SiameseNetwork

def load_model(model_path, device):
    model_net = SiameseNetwork().to(device)
    model_net.load_state_dict(torch.load(model_path, map_location=device))
    model_net.eval()
    return model_net

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="siamese_model.pth", help="学習済みモデルのパス")
    parser.add_argument("--fingerprint", type=str, required=True, help="指紋画像のパス")
    parser.add_argument("--name", type=str, required=True, help="登録する名前またはID")
    parser.add_argument("--db_path", type=str, default="fingerprint_db.pkl", help="指紋データベースのファイルパス")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_net = load_model(args.model_path, device)

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])

    img = Image.open(args.fingerprint).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        # 特徴抽出後に正規化を実施
        features = model_net.extract_features(img_tensor)
    features = features.cpu().numpy()

    # 既存のデータベースがあれば読み込み、なければ新規作成
    if os.path.exists(args.db_path):
        with open(args.db_path, "rb") as f:
            db = pickle.load(f)
    else:
        db = {}

    db[args.name] = features
    with open(args.db_path, "wb") as f:
        pickle.dump(db, f)
    print(f"指紋 {args.fingerprint} を {args.name} として登録しました。")

if __name__ == "__main__":
    main()
