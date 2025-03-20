from torch.utils.data import DataLoader
from torchvision import transforms
from datasetloader_v2 import SiameseFingerprintDatasetV2

def main():
    
    # transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor()
    ])
    root_dir = './NIST300/SD300a/images'
    
    dataset = SiameseFingerprintDatasetV2(root_dir=root_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    for batch_idx, (img1, img2, labels) in enumerate(dataloader):
        print(f"バッチ {batch_idx + 1}:")
        print("画像1の形状:", img1.shape)
        print("画像2の形状:", img2.shape)
        print("ラベル:", labels)
    return

if __name__ == '__main__':
    main()