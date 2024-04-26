import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from timm.models.layers.helpers import to_2tuple
import timm
import torch.nn as nn
import timm.models.swin_transformer
import cv2
import os
import h5py
import argparse



class ConvStem(nn.Module):

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()

        assert patch_size == 4
        assert embed_dim % 8 == 0

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0],
                          img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        stem = []
        input_dim, output_dim = 3, embed_dim // 8
        for l in range(2):
            stem.append(nn.Conv2d(input_dim, output_dim,
                        kernel_size=3, stride=2, padding=1, bias=False))
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


def ctranspath():
    model = timm.create_model(
        'swin_tiny_patch4_window7_224', embed_layer=ConvStem, pretrained=False)
    return model


mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
trnsfrms_val = transforms.Compose(
    [
        # transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]
)


class roi_dataset(Dataset):
    def __init__(self, imgs,normalizer=None):
        super().__init__()
        self.transform = trnsfrms_val

        self.images_lst = imgs
        self.normalizer=normalizer

    def __len__(self):
        return len(self.images_lst)

    def __getitem__(self, idx):
        path = self.images_lst[idx]
        if self.normalizer is not None:
            try:
                image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
                image = self.normalizer.transform(image)
                image = self.transform(image)
            except:
                image=None
        else:
            image = Image.open(path).convert('RGB')
            image = self.transform(image)

        return image


class extract_path_features(object):
    def __init__(self, checkpoint):
        model = ctranspath()
        model.head = nn.Identity()
        td = torch.load(checkpoint)
        model.load_state_dict(td['model'], strict=True)
        model.eval()
        self.model = model.cuda()
    def auto_extract(self,tile_path:str,save_path:str,batch_size:int):
        tile_file=os.listdir(tile_path)
        tile_case=[]
        for tc in  tile_file:
            if tc.startswith("C"):
                tile_case.append(tc[:9])
            elif tc.startswith("T"):
                tile_case.append(tc[:12])
            else:
                tile_case.append(tc[:20])
        tile_case=set(tile_case)
        for case in tile_case:
            img_file=[]
            if os.path.exists(os.path.join(save_path,f"{case}.h5")):
                continue
            for tf in tile_file:
                if tf.startswith(case):
                    imgs=os.listdir(os.path.join(tile_path,tf))
                    imgs=[os.path.join(tile_path,tf,x) for x in imgs]
                    img_file.extend(imgs)
            data = roi_dataset(imgs=img_file)
            database_loader = torch.utils.data.DataLoader(
                data, batch_size=batch_size, shuffle=False)
            npy_save =np.empty(shape=(0,768))
            with torch.no_grad():
                for batch in tqdm(database_loader):
                    if batch is not None:
                        batch=batch.cuda()
                        features = self.model(batch)
                        features = features.cpu().numpy()
                        npy_save=np.append(npy_save,features,axis=0)
            if npy_save.shape[0]<1:
                continue
            with h5py.File(os.path.join(save_path,f"{case}.h5"), 'w') as hf:
                hf.create_dataset("data",  data=torch.FloatTensor(npy_save))
            print(f"Extract {npy_save.shape} features.")
            

def get_extractor(ckpt):
    model = ctranspath()
    model.head = nn.Identity()
    td = torch.load(ckpt)
    model.load_state_dict(td['model'], strict=True)
    model.eval()
    model = model.cuda()
    return model


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='Patch extraction for WSI')
    parser.add_argument('--extract_pretrain_path', type=str, default="", help='ctranspath.pth')
    parser.add_argument('-batch_size',  type=int, default=256,help='batch size')
    parser.add_argument('-tile_path', type=str, default="",help='Tiled Cancer patches dir')
    parser.add_argument('-save_path',  type=str, default="",help='Saving .h5 dir')
    args = parser.parse_args()
    
    extract = extract_path_features(checkpoint=args.extract_pretrain_path)
    extract.auto_extract(tile_path=args.tile_path,save_path=args.save_path,batch_size=args.batch_size)
