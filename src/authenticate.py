import torch
from torchvision import transforms
from PIL import Image
import argparse
import os
import pickle
import numpy as np

from model import SiameseNetwork

def load_model(model_path, device):
    model_net = SiameseNetwork().to(device)
    model_net.load_state_dict(torch.load(model_path, map_location=device))
    model_net.eval()
    return model_net

def cosine_similarity(vec1, vec2):
    # 余弦類似度を計算（値は -1～1、1 に近いほど類似）
    dot = np.dot(vec1, vec2)
    norm = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return dot / norm if norm != 0 else 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="siamese_model.pth", help="学習済みモデルのパス")
    parser.add_argument("--fingerprint", type=str, required=True, help="認証する指紋画像のパス")
    parser.add_argument("--db_path", type=str, default="fingerprint_db.pkl", help="指紋データベースのファイルパス")
    parser.add_argument("--threshold", type=float, default=0.7, help="類似度の閾値（cosine similarity）")
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
        # 特徴抽出後、正規化済みの特徴量を取得
        query_features = model_net.extract_features(img_tensor)
    query_features = query_features.cpu().numpy().flatten()

    if not os.path.exists(args.db_path):
        print("指紋データベースが見つかりません。")
        return
    with open(args.db_path, "rb") as f:
        db = pickle.load(f)

    best_match = None
    best_score = -1.0  # cosine similarity は -1～1 の値（高いほど類似）
    for name, features in db.items():
        db_feat = features.flatten()
        cos_sim = cosine_similarity(query_features, db_feat)
        print(f"{name} との cosine similarity: {cos_sim:.4f}")
        if cos_sim > best_score:
            best_score = cos_sim
            best_match = name

    if best_score >= args.threshold:
        print(f"認証成功: {best_match}（cosine similarity: {best_score:.4f}）")
    else:
        print("認証失敗: 一致する指紋が見つかりません。")

if __name__ == "__main__":
    main()
