import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import glob
from mae_utils import PURPLE, YELLOW, imagenet_mean, imagenet_std


class DatasetEnhance(Dataset):
    def __init__(self, datapath, image_transform, mask_transform, padding: bool = 1,
                 use_original_imgsize: bool = False, flipped_order: bool = False,
                 reverse_support_and_query: bool = False, random: bool = False, trn: bool = False):
        self.padding = padding
        self.random = random
        self.use_original_imgsize = use_original_imgsize
        self.image_transform = image_transform
        self.reverse_support_and_query = reverse_support_and_query
        self.mask_transform = mask_transform
        self.trn = trn
        if trn:
            self.img_src_dir = "../datasets/light_enhance/our485/low/"
        else:
            self.img_src_dir = "../datasets/light_enhance/eval15/low/"
        self.img_path_list = glob.glob(self.img_src_dir + "*.png")
        self.flipped_order = flipped_order
        np.random.seed(5)
        if trn:
            self.indices = np.random.choice(np.arange(0, len(self.img_path_list)-1), size=16, replace=False)
        else:
            self.indices = np.random.choice(np.arange(0, len(self.img_path_list)-1), size=100, replace=True)
        # self.img_metadata_classwise = self.build_img_metadata_classwise()
        print()


    def __len__(self):
        if self.trn:
            return 16
        else:
            return 100

    def create_grid_from_images(self, support_img, support_mask, query_img, query_mask):
        if self.reverse_support_and_query:
            support_img, support_mask, query_img, query_mask = query_img, query_mask, support_img, support_mask
        canvas = torch.ones((support_img.shape[0], 2 * support_img.shape[1] + 2 * self.padding,
                             2 * support_img.shape[2] + 2 * self.padding))
        canvas[:, :support_img.shape[1], :support_img.shape[2]] = support_img
        if self.flipped_order:
            canvas[:, :support_img.shape[1], -support_img.shape[2]:] = query_img
            canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = query_mask
            canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = support_mask
        else:
            canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = query_img
            canvas[:, :support_img.shape[1], -support_img.shape[2]:] = support_mask
            canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = query_mask

        return canvas

    def __getitem__(self, idx):
        support_idx = np.random.choice(np.arange(0, len(self)-1))
        query_img = self.img_path_list[self.indices[idx]]
        support_img = self.img_path_list[self.indices[support_idx]]
        query_trg = query_img.replace('low', 'high')
        support_trg = support_img.replace('low', 'high')
        # query, support = self.ds[idx], self.ds[support_idx]
        # support_idx = np.random.choice(self.img_metadata_classwise[query[1]], 1, replace=False)[0]
        # support = np.random.choice(self.img_metadata_classwise[query], 1, replace=False)
        query_img = Image.open(query_img).convert('RGB')
        support_img = Image.open(support_img).convert('RGB')
        query_trg = Image.open(query_trg)
        support_trg = Image.open(support_trg)
        # a = np.array(support_trg)
        # b = np.array(support_img)
        # c = np.array(query_img)
        query_img, query_trg = self.image_transform(query_img), self.mask_transform(query_trg) 
        support_img, support_trg = self.image_transform(support_img), self.mask_transform(support_trg) 
        grid = self.create_grid_from_images(support_img, support_trg, query_img, query_trg)
        # grid = grid / 255.
        # a = np.array(support_trg)
        # b = np.array(support_img)
        # c = np.array(query_img)
        # d = np.array(grid)
        grid = ((grid - imagenet_mean[:, None, None]) / imagenet_std[:, None, None]).float()
        # d = np.array(grid)
        return grid
    
    def build_img_metadata_classwise(self):
        img_metadata_classwise = {}
        for class_id in range(1000):
            img_metadata_classwise[class_id] = []
        
        for i in range(len(self.ds.imgs)):
            _, img_class = self.ds.imgs[i]
            img_metadata_classwise[img_class] += [i]

        return img_metadata_classwise