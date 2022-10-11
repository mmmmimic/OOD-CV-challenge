from torch.utils.data import Dataset
import pandas as pd
from os.path import join
from torchvision import transforms
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from aug_mix import aug
from imgaug import augmenters as iaa
from copy import deepcopy
import torchvision
import cv2

# 对遮挡，天气的表现不太好。
# - 添加cutmix，模拟遮挡的情况
# - 添加channel shuffle，模拟天气不好的情况

# cutmix
def rand_bbox(size, lamb):
    W = size[0]
    H = size[1]
    cut_rat = np.sqrt(1. - lamb)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def saliency_bbox(img, lam):
    size = img.size()
    W = size[1]
    H = size[2]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # initialize OpenCV's static fine grained saliency detector and compute the saliency map
    temp_img = img.cpu().numpy().transpose(1, 2, 0)
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliencyMap) = saliency.computeSaliency(temp_img)
    saliencyMap = (saliencyMap * 255).astype("uint8")

    maximum_indices = np.unravel_index(np.argmax(saliencyMap, axis=None), saliencyMap.shape)
    x = maximum_indices[0]
    y = maximum_indices[1]

    bbx1 = np.clip(x - cut_w // 2, 0, W)
    bby1 = np.clip(y - cut_h // 2, 0, H)
    bbx2 = np.clip(x + cut_w // 2, 0, W)
    bby2 = np.clip(y + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

class PoseData(Dataset):
    def __init__(self, data_folder, transforms, split='test') -> None:
        """
        the data folder should look like
        - datafolder
            - Images
            - labels.csv        
        """
        super().__init__()
        csv_dir = join(data_folder, 'labels.csv')
        self.csv = pd.read_csv(csv_dir)
        self.folder = data_folder
        self.tfs = transforms
        self.split = split
        self.name_list = [
            'aeroplane', 
            'bicycle',
            'boat',
            'bus',
            'car',
            'chair', 
            'diningtable',
            'motorbike',
            'sofa',
            'train'
            ]
        
    def __len__(self):
        return len(self.csv)
    
    def __getitem__(self, index):
        image_dir = self.csv['imgs'][index]
        zaimuth = self.csv['azimuth'][index]
        label = self.csv['labels'][index]
        image_dir = join(self.folder, 'Images', label, image_dir)
        image = Image.open(image_dir)
        if self.split == 'train' and np.random.rand() < 0.5:
            seq = iaa.Sequential(
                [
                iaa.OneOf([
                    iaa.Rain(drop_size=(0.2, 0.4), speed=(0.01, 0.02)),#下雨
                   iaa.imgcorruptlike.Snow(severity=1),#下雪
                   iaa.imgcorruptlike.Fog(severity=1)#雾化
                   ]),
                ]
            )
            try:
                image = seq.augment_image(np.asarray(deepcopy(image)))
                image = Image.fromarray(image)
            except AssertionError:
                pass
        if self.split == 'train':
            image = aug(image, self.tfs)
            # rotation
            i = np.random.randint(0, 4)
            if i == 1:
                image = torchvision.transforms.functional.rotate(image, 90)
            elif i == 2:
                image = torchvision.transforms.functional.rotate(image, 180)
            elif i == 3:
                image = torchvision.transforms.functional.rotate(image, 270)
            angle = i
            
        else:
            image = self.tfs(image)
            angle = 0
        
        label = self.name_list.index(label)
        
        
        data = {
            'image_dir': self.csv['imgs'][index],
            'image': image,
            'label': label,
            'zaimuth': zaimuth/360.,
            'angle': angle,
        }
        
        return data

if __name__ == "__main__":
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    tfs = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            normalize,
        ])
    data = PoseData('ROBINv1.1-cls-pose/train/', tfs, 'train')
    d = data[1000]
    print(d['image'].shape)
    plt.figure()
    plt.imshow(d['image'].permute(1,2,0))
    plt.show()
    plt.savefig('test.png')
    print(d['angle'])
    
    print(d['zaimuth'], d['label'])
    