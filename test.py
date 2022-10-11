from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
from glob import glob
from pathlib import PurePath
import numpy as np
import timm
import torchvision
import time

img_list = glob('phase2-cls/images/*.jpg')

name_list = [
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

class PoseData(Dataset):
    def __init__(self, transforms) -> None:
        """
        the data folder should look like
        - datafolder
            - Images
            - labels.csv        
        """
        super().__init__()
        self.img_list = glob('phase2-cls/images/*.jpg')
        self.img_list = sorted(self.img_list, key=lambda x: eval(PurePath(x).parts[-1][:-4]))
        self.trs = transforms
        
        
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        image_dir = self.img_list[index]
        image_name = PurePath(image_dir).parts[-1]
        image = Image.open(image_dir)
        image = self.trs(image)
        
        
        return image, image_name

if __name__ == "__main__":
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    tfs = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])
    
    model1 = timm.models.swin_base_patch4_window7_224(pretrained=True, num_classes=15)
    model1 = torch.nn.DataParallel(model1)
    model1.load_state_dict(torch.load('swin15_best.pth.tar')['state_dict'])
    model1 = model1.cuda()
    model1.eval()
    
    model2 = timm.models.convnext_base(pretrained=True, num_classes=15)
    model2 = torch.nn.DataParallel(model2)
    model2.load_state_dict(torch.load('convnext15_best.pth.tar')['state_dict'])
    model2 = model2.cuda()
    model2.eval()

    dataset = PoseData(tfs)
    loader = DataLoader(dataset, batch_size=128, shuffle=False, drop_last=False, num_workers=4)
    
    image_dir = []
    preds = []
    for image, pth, rot, angle, sc in loader:
        image_dir.append(list(pth))
        image = image.cuda()
            
        with torch.no_grad():
                    
            model1.eval()
            pred1 = model1(image)
            model2.eval()
            pred2 = model2(image)
            
            entropy1 = -torch.sum(torch.softmax(pred1[:,:10], dim=1)*torch.logsoftmax(pred1[:,:10], dim=1), dim=-1, keep_dim=True)
            entropy2 = -torch.sum(torch.softmax(pred2[:,:10], dim=1)*torch.logsoftmax(pred2[:,:10], dim=1), dim=-1, keep_dim=True)
            entropy = entropy1 + entropy2
         
            pred = torch.softmax(pred1[:,:10], dim=1)*(entropy - entropy1)/entropy  + torch.softmax(pred2[:,:10], dim=1)*(entropy - entropy2)/entropy
            pred = torch.argmax(pred[:,:10], dim=1)
            p = []
            for i in range(pred.size(0)):
                p.append(name_list[pred[i].item()])
        p = np.array(p)
        preds.append(p)
        print(len(np.concatenate(preds)))
        
    image_dir = np.array(sum(image_dir, []))
    preds = np.concatenate(preds)
    
    csv = {'imgs':np.array(image_dir), 'pred':np.array(preds), 
        }
    csv = pd.DataFrame(csv)
    print(csv)
    
    csv.to_csv('results.csv', index=False)
        