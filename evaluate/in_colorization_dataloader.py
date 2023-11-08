import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from mae_utils import imagenet_mean, imagenet_std


class DatasetColorization(Dataset):
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
            self.ds = ImageFolder(os.path.join(datapath, 'train'))
        else:
            self.ds = ImageFolder(os.path.join(datapath, 'val'))
        self.flipped_order = flipped_order
        np.random.seed(5)
        if trn:
            self.indices = np.random.choice(np.arange(0, len(self.ds)-1), size=320, replace=False)
        else:
            self.indices = np.random.choice(np.arange(0, len(self.ds)-1), size=1000, replace=False)
        self.img_metadata_classwise = self.build_img_metadata_classwise()
        print()


    def __len__(self):
        if self.trn:
            return 320
        else:
            return 1000

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
        # support_idx = np.random.choice(np.arange(0, len(self)-1))
        idx = self.indices[idx]
        # query, support = self.ds[idx], self.ds[support_idx]
        query = self.ds[idx]
        support_idx = np.random.choice(self.img_metadata_classwise[query[1]], 1, replace=False)[0]
        support = self.ds[support_idx]
        # support = np.random.choice(self.img_metadata_classwise[query], 1, replace=False)
        query_img, query_mask = self.mask_transform(query[0]), self.image_transform(query[0])
        support_img, support_mask = self.mask_transform(support[0]), self.image_transform(support[0])
        grid = self.create_grid_from_images(support_img, support_mask, query_img, query_mask)
        grid = ((grid - imagenet_mean[:, None, None]) / imagenet_std[:, None, None]).float()
        batch = {'query_img': query_img, 'query_mask': query_mask, 'support_img': support_img,
                 'support_mask': support_mask, 'grid': grid}

        return batch['grid']
    
    def build_img_metadata_classwise(self):
        img_metadata_classwise = {}
        for class_id in range(1000):
            img_metadata_classwise[class_id] = []
        
        for i in range(len(self.ds.imgs)):
            _, img_class = self.ds.imgs[i]
            img_metadata_classwise[img_class] += [i]

        return img_metadata_classwise