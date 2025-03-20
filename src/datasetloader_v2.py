import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class SiameseFingerprintDatasetV2(Dataset):
    """
    (subject, frgp) が同じ画像を正例ペア (label=0) とし、
    それ以外を負例ペア (label=1) として返すカスタムデータセット。

    ディレクトリ構造の例:
    images/
      ├─ 500/
      │   ├─ plain/
      │   │   ├─ 00001000_plain_500_01.png
      │   │   └─ ...
      │   ├─ roll/
      │   │   ├─ 00001000_roll_500_01.png
      │   │   └─ ...
      ├─ 1000/
      │   ├─ plain/
      │   └─ roll/
      ├─ 2000/
      │   ├─ plain/
      │   └─ roll/
      └─ ... (他の解像度)

    ファイル名: SUBJECT_IMPRESSION_PPI_FRGP.EXT
      SUBJECT: 被験者ID (例: "00001000")
      IMPRESSION: plain/roll など
      PPI: 500/1000/2000 など
      FRGP: 指のポジション (例: "01" は右手親指など)
      EXT: png/jpg など
    """

    def __init__(self, root_dir, transform=None, exts=(".png", ".jpg", ".jpeg", ".bmp")):
        """
        Args:
            root_dir (str): 画像が格納された最上位ディレクトリ (例: "images")
            transform (callable): 画像前処理 (torchvision.transforms など)
            exts (tuple): 読み込む拡張子のタプル
        """
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.exts = exts

        # 全画像のメタデータを保持するリスト
        # 例: [(fullpath, subject, frgp), ...]
        self.metadata_list = []

        # (subject, frgp) -> [メタデータのインデックス, ...]
        self.group_dict = {}

        # os.walk() でサブフォルダを含めて探索
        for dirpath, dirnames, filenames in os.walk(root_dir):
            for fname in filenames:
                # checksum_xxx.csv や segmentation_yyy.csv などはスキップ
                if not fname.lower().endswith(self.exts):
                    continue

                full_path = os.path.join(dirpath, fname)
                # ファイル名から subject, impression, ppi, frgp をパース
                # 例: "00001000_roll_500_01.png" -> ["00001000", "roll", "500", "01.png"]
                base = os.path.splitext(fname)[0]  # "00001000_roll_500_01"
                parts = base.split("_")
                if len(parts) < 4:
                    # 想定外のファイル名はスキップするか、例外を投げる
                    continue

                subject = parts[0]         # "00001000"
                # impression = parts[1]   # "roll" or "plain" (必要なら使う)
                # ppi = parts[2]          # "500" or "1000" etc. (必要なら使う)
                frgp = parts[3]           # "01" など

                # メタデータを追加
                meta_index = len(self.metadata_list)
                self.metadata_list.append((full_path, subject, frgp))

                # (subject, frgp) をキーにしてインデックスを追加
                group_key = (subject, frgp)
                if group_key not in self.group_dict:
                    self.group_dict[group_key] = []
                self.group_dict[group_key].append(meta_index)

        # (subject, frgp) のリスト（負例ペア生成で他グループを選ぶときに使う）
        self.group_keys = list(self.group_dict.keys())
        # メタデータがソートされていない場合もあるので、必要に応じて並び替えても良い

    def __len__(self):
        return len(self.metadata_list)

    def __getitem__(self, idx):
        """
        1枚目の画像を self.metadata_list[idx] とし、
        正例 or 負例ペアをランダムに生成して返す。
        """
        img1_path, subject1, frgp1 = self.metadata_list[idx]
        group_key1 = (subject1, frgp1)

        # 正例（same = True）か負例（same = False）かをランダムに選ぶ
        same = random.choice([True, False])

        if same:
            # 同じ (subject, frgp) のグループから別の画像を選択
            indices = self.group_dict[group_key1]
            if len(indices) < 2:
                # グループ内に1枚しか無い場合はそのまま同じ画像を返す（あまり望ましくはない）
                img2_idx = idx
            else:
                img2_idx = idx
                while img2_idx == idx:
                    img2_idx = random.choice(indices)
            label = 0  # 同一人物かつ同一指
        else:
            # 違うグループ (subject, frgp) をランダムに選ぶ
            group_key2 = group_key1
            while group_key2 == group_key1:
                group_key2 = random.choice(self.group_keys)
            img2_idx = random.choice(self.group_dict[group_key2])
            label = 1  # 異なる人物 or 異なる指（もしくはその両方）

        # 2枚目の画像のパスを取得
        img2_path, subject2, frgp2 = self.metadata_list[img2_idx]

        # 画像読み込み & 前処理
        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(label, dtype=torch.float32)