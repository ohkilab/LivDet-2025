import torch
import torch.nn as nn
import torchvision.models as models

class SiameseNetwork(nn.Module):
    def __init__(self, feature_dim=2048):
        super(SiameseNetwork, self).__init__()
        # torchvision の resnext_101 を読み込み、最終層（全結合層）を Identity に置換
        self.feature_extractor = models.resnext101_32x8d(pretrained=True)
        self.feature_extractor.fc = nn.Identity()

    def forward(self, img1, img2):
        # 2 枚の画像から特徴を抽出
        feat1 = self.feature_extractor(img1)
        feat2 = self.feature_extractor(img2)
        return feat1, feat2

    def extract_features(self, img):
        # 1 枚の画像から特徴抽出（抽出後、正規化も実施）
        feat = self.feature_extractor(img)
        feat_norm = feat / feat.norm(dim=1, keepdim=True)
        return feat_norm
