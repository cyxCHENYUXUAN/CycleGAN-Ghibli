import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torchvision

class RealGhibliDataset(Dataset):
    def __init__(self, ghibli_dir=None, real_dir=None, transform=None):
        self.ghibli_dir = ghibli_dir
        self.real_dir = real_dir
        self.transform = transform
        self.ghibli_files = sorted(os.listdir(self.ghibli_dir))
        self.real_files = sorted(os.listdir(self.real_dir))

    def __len__(self):
        return max(len(self.ghibli_files), len(self.real_files))

    def __getitem__(self, idx):
        ghibli_img = Image.open(os.path.join(self.ghibli_dir, self.ghibli_files[idx % len(self.ghibli_files)])).convert("RGB")
        real_img = Image.open(os.path.join(self.real_dir, self.real_files[idx % len(self.real_files)])).convert("RGB")
        if self.transform:
            ghibli_img = self.transform(ghibli_img)
            real_img = self.transform(real_img)
        return real_img, ghibli_img
    
# if __name__ == '__main__':
 
#     data_transforms = transforms.Compose([
#         transforms.Resize((128, 128)),
#         transforms.RandomHorizontalFlip(),  # 随机水平翻转
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # 将图片像素值归一化到[-1, 1]
#     ])

#     GHIBLI_DIR = "../Data/dataset/trainB_ghibli"
#     REAL_DIR = "../Data/dataset/trainA"
 
#     train_dataset = RealGhibliDataset(
#         ghibli_dir=GHIBLI_DIR,
#         real_dir=REAL_DIR,
#         transform=data_transforms
#     )
    
#     #在Windows上，num_workers>0必须放在if __name__ == '__main__':
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=1,
#         shuffle=True,
#         num_workers=2,
#         drop_last=True
#     )
 
#     def imshow(tensor_grid, title):
#         img = tensor_grid / 2 + 0.5
#         npimg = img.numpy()
#         plt.figure(figsize=(10, 5))
#         plt.imshow(np.transpose(npimg, (1, 2, 0)))
#         plt.title(title)
#         plt.axis('off')
#         plt.show()
 
#     print("正在加载图片")
#     data_iterator = iter(train_loader)
#     real_images_batch, ghibli_images_batch = next(data_iterator)
 
#     comparison_grid = torchvision.utils.make_grid(
#         torch.cat((real_images_batch, ghibli_images_batch), 0), 
#         nrow=real_images_batch.shape[0]
#     )
    
#     imshow(comparison_grid,title=None)