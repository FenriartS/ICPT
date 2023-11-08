"""Based on https://github.com/Seokju-Cho/Volumetric-Aggregation-Transformer/blob/main/data/pascal.py
"""
import os
from PIL import Image
from scipy.io import loadmat
import numpy as np
import torch
from torch.utils.data import Dataset
from .mae_utils import PURPLE, YELLOW, imagenet_mean, imagenet_std
import random
# from .mae_utils import PURPLE, YELLOW
# from evaluate.fda import FDA_source_to_target

def create_grid_from_images_old(canvas, support_img, support_mask, query_img, query_mask):
    canvas[:, :support_img.shape[1], :support_img.shape[2]] = support_img
    canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = query_img
    canvas[:, :support_img.shape[1], -support_img.shape[2]:] = support_mask
    canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = query_mask
    return canvas

class DatasetPASCAL(Dataset):
    def __init__(self, datapath, fold, image_transform, mask_transform, padding: bool = 1, use_original_imgsize: bool = False, flipped_order: bool = False,
                 reverse_support_and_query: bool = False, random: bool = False, ensemble: bool = False, purple: bool = False, trn: bool = False):
        self.fold = fold
        self.nfolds = 4
        self.flipped_order = flipped_order
        self.nclass = 20
        self.padding = padding
        self.random = random
        self.ensemble = ensemble
        self.purple = purple
        self.use_original_imgsize = use_original_imgsize
        self.trn = trn

        self.img_path = os.path.join(datapath, 'VOCdevkit/VOC2012/JPEGImages/')
        self.ann_path = os.path.join(datapath, 'VOCdevkit/VOC2012/SegmentationClassAug/')
        self.image_transform = image_transform
        self.reverse_support_and_query = reverse_support_and_query
        self.mask_transform = mask_transform

        self.class_ids = self.build_class_ids()
        self.img_metadata = self.build_img_metadata()
        self.img_metadata_classwise = self.build_img_metadata_classwise()

        self.num_shots = 16
        if self.trn and self.num_shots > 0:
            self.build_fewshot_metadata()

    def __len__(self):
        if self.trn:
            return len(self.img_metadata)
        return 1000

    def create_grid_from_images(self, support_img, support_mask, query_img, query_mask, flip: bool = False):
        if self.reverse_support_and_query:
            support_img, support_mask, query_img, query_mask = query_img, query_mask, support_img, support_mask
        canvas = torch.ones((support_img.shape[0], 2 * support_img.shape[1] + 2 * self.padding, 2 * support_img.shape[2] + 2 * self.padding))
        canvas[:, :support_img.shape[1], :support_img.shape[2]] = support_img

        # canvas = torch.ones(3, 224, 224)
        # canvas[:, 16:16+63, 16:16+63] = support_img
        # canvas[:, 16:16+63, 16+64:16+64+63] = support_mask
        # canvas[:, 16:16+63, 16+64+64:16+64+64+63] = support_img
        # canvas[:, 16+64:16+64+63, 16:16+63] = query_img
        # canvas[:, 16+64:16+64+63, 16+64:16+64+63] = query_mask
        # canvas[:, 16+64:16+64+63, 16+64+64:16+64+64+63] = query_img
        # canvas[:, 16+64+64:16+64+64+63, 16:16+63] = support_img2
        # canvas[:, 16+64+64:16+64+64+63, 16+64:16+64+63] = support_mask2
        # canvas[:, 16+64+64:16+64+64+63, 16+64+64:16+64+64+63] = support_img2

        if flip:
            canvas[:, :support_img.shape[1], -support_img.shape[2]:] = query_img
            canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = query_mask
            canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = support_mask
        else:
            canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = query_img
            canvas[:, :support_img.shape[1], -support_img.shape[2]:] = support_mask
            canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = query_mask

        return canvas

    def __getitem__(self, idx):
        if not self.trn:
            idx %= len(self.img_metadata)  # for testing, as n_images < 1000
        query_name, support_name, class_sample_query, class_sample_support = self.sample_episode(idx)
        query_img, query_cmask, support_img, support_cmask, org_qry_imsize = self.load_frame(query_name, support_name)
        # query_name, support_name2, class_sample_query, class_sample_support = self.sample_episode(idx)
        # query_img, query_cmask, support_img, support_cmask, support_img2, support_cmask2, org_qry_imsize = self.load_frame(query_name, support_name ,support_name2)

        if self.image_transform:
            query_img = self.image_transform(query_img)
        query_mask, query_ignore_idx = self.extract_ignore_idx(query_cmask, class_sample_query, purple=self.purple)
        if self.mask_transform:
            query_mask = self.mask_transform(query_mask)
                
        if self.image_transform:
            support_img = self.image_transform(support_img)
            # support_img2 = self.image_transform(support_img2)
        support_mask, support_ignore_idx = self.extract_ignore_idx(support_cmask, class_sample_support, purple=self.purple)
        # support_mask2, support_ignore_idx2 = self.extract_ignore_idx(support_cmask2, class_sample_support, purple=self.purple)
        if self.mask_transform:
            support_mask = self.mask_transform(support_mask)
            # support_mask2 = self.mask_transform(support_mask2)
        grid = self.create_grid_from_images(support_img, support_mask, query_img, query_mask, flip=self.flipped_order)
        grid = ((grid - imagenet_mean[:, None, None]) / imagenet_std[:, None, None]).float()
        # support_img = FDA_source_to_target(support_img.unsqueeze(0), query_img.unsqueeze(0), L=0.02).squeeze(0)
        # support_img2 = FDA_source_to_target(support_img2.unsqueeze(0), query_img.unsqueeze(0), L=0.03).squeeze(0)
        # grid = self.create_grid_from_images(support_img, support_mask, support_img2, support_mask2, query_img, query_mask, flip=self.flipped_order)
        # mask_loc = torch.where(support_mask==1)
        # up, down, left, right = min(mask_loc[1]), max(mask_loc[1]), min(mask_loc[2]), max(mask_loc[2])
        # support_mask_coarse = torch.zeros_like(support_mask)
        # support_mask_coarse[:, up:down+1, left:right+1] = 1
        # grid_coarse = self.create_grid_from_images(support_img, support_mask_coarse, query_img, query_mask, flip=self.flipped_order)
        if self.ensemble:
            grid2 = self.create_grid_from_images(support_img, support_mask, query_img, query_mask, (not self.flipped_order))


            support_purple_mask, _ = self.extract_ignore_idx(support_cmask, class_sample_support,
                                                                       purple=True)
            if self.mask_transform:
                support_purple_mask = self.mask_transform(support_purple_mask)

            grid3 = self.create_grid_from_images(support_img, support_purple_mask, query_img, query_mask,
                                                flip=self.flipped_order)

            grid4 = self.create_grid_from_images(support_img, support_purple_mask, query_img, query_mask,
                                                flip=(not self.flipped_order))

            grid = grid, grid2, grid3, grid4
        # batch = {'query_img': query_img,
        #          'query_mask': query_mask,
        #          'query_name': query_name,
        #          'query_ignore_idx': query_ignore_idx,
        #          'org_query_imsize': org_qry_imsize,
        #          'support_img': support_img,
        #          'support_mask': support_mask,
        #          'support_name': support_name,
        #          'support_ignore_idx': support_ignore_idx,
        #          'class_id': torch.tensor(class_sample_query),
        #         #  'grid_coarse': grid_coarse,
        #          'grid': grid}

        # return batch
        return grid

    def extract_ignore_idx(self, mask, class_id, purple):
        mask = np.array(mask)
        boundary = np.floor(mask / 255.)
        
        if not purple:
            mask[mask != class_id + 1] = 0
            mask[mask == class_id + 1] = 255
            return Image.fromarray(mask), boundary
        color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for x in range(mask.shape[0]):
            for y in range(mask.shape[1]):
                if mask[x,y] != class_id + 1:
                    color_mask[x, y] = np.array(PURPLE)
                else:
                    color_mask[x, y] = np.array(YELLOW)
        return Image.fromarray(color_mask), boundary


    def load_frame(self, query_name, support_name):
        query_img = self.read_img(query_name)
        query_mask = self.read_mask(query_name)
        support_img = self.read_img(support_name)
        support_mask = self.read_mask(support_name)
        # support_img2 = self.read_img(support_name2)
        # support_mask2 = self.read_mask(support_name2)
        org_qry_imsize = query_img.size

        return query_img, query_mask, support_img, support_mask, org_qry_imsize
        # return query_img, query_mask, support_img, support_mask, support_img2, support_mask2, org_qry_imsize

    def read_mask(self, img_name):
        r"""Return segmentation mask in PIL Image"""
        mask = Image.open(os.path.join(self.ann_path, img_name) + '.png')
        return mask

    def read_img(self, img_name):
        r"""Return RGB image in PIL Image"""
        return Image.open(os.path.join(self.img_path, img_name) + '.jpg')

    def sample_episode(self, idx):
        """Returns the index of the query, support and class."""
        query_name, class_sample = self.img_metadata[idx]
        if not self.random:
            support_class = class_sample
        else: 
            support_class = np.random.choice([k for k in self.img_metadata_classwise.keys() if self.img_metadata_classwise[k]], 1, replace=False)[0]
        while True:  # keep sampling support set if query == support
            support_name = np.random.choice(self.img_metadata_classwise[support_class], 1, replace=False)[0]
            if query_name != support_name: 
                break
        return query_name, support_name, class_sample, support_class

    def build_class_ids(self):
        if self.trn:
            class_ids_val = [i for i in range(self.nclass)]
        else:
            nclass_trn = self.nclass // self.nfolds
            class_ids_val = [self.fold * nclass_trn + i for i in range(nclass_trn)]
        return class_ids_val

    def build_img_metadata(self):

        def read_metadata(split, fold_id):
            cwd = os.path.dirname(os.path.abspath(__file__))
            fold_n_metadata = os.path.join(cwd, 'splits/pascal/%s/fold%d.txt' % (split, fold_id))
            with open(fold_n_metadata, 'r') as f:
                fold_n_metadata = f.read().split('\n')[:-1]
            fold_n_metadata = [[data.split('__')[0], int(data.split('__')[1]) - 1] for data in fold_n_metadata]
            return fold_n_metadata

        img_metadata = []
        if self.trn:
            for i in range(self.nfolds):
                img_metadata += read_metadata('trn', i)
            print('Total (trn) images are : %d' % len(img_metadata))
        else:
            for i in range(self.nfolds):
                img_metadata += read_metadata('val', i)
            # img_metadata = read_metadata('val', self.fold)       
            print('Total (val) images are : %d' % len(img_metadata))

        return img_metadata

    def build_img_metadata_classwise(self):
        img_metadata_classwise = {}
        for class_id in range(self.nclass):
            img_metadata_classwise[class_id] = []

        for img_name, img_class in self.img_metadata:
            img_metadata_classwise[img_class] += [img_name]
        return img_metadata_classwise
    
    def build_fewshot_metadata(self):
        fewshot_img_metadata = []
        fewshot_img_metadata_classwise = {}
        for label, items in self.img_metadata_classwise.items():
            sample_items = random.sample(items, self.num_shots)
            fewshot_img_metadata_classwise[label] = sample_items
            fewshot_img_metadata.extend([[item, label] for item in sample_items])
        self.img_metadata = fewshot_img_metadata
        self.img_metadata_classwise = fewshot_img_metadata_classwise

class DatasetPASCALforFinetune(Dataset):
    def __init__(self, datapath, fold, image_transform, mask_transform, num_supports=1, padding: bool = 1,
                 use_original_imgsize: bool = False, random: bool = False):
        self.fold = fold
        self.nfolds = 4
        self.nclass = 20
        self.padding = padding
        self.random = random
        self.use_original_imgsize = use_original_imgsize

        self.img_path = os.path.join(datapath, 'VOCdevkit/VOC2012/JPEGImages/')
        self.ann_path = os.path.join(datapath, 'VOCdevkit/VOC2012/SegmentationClassAug/')
        self.image_transform = image_transform
        self.mask_transform = mask_transform

        self.class_ids = self.build_class_ids()
        self.img_metadata = self.build_img_metadata()
        self.img_metadata_classwise = self.build_img_metadata_classwise()
        self.num_supports = num_supports

    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        idx %= len(self.img_metadata)  # for testing, as n_images < 1000
        query_name, support_names, class_sample_query, class_sample_support = self.sample_episode(idx)
        query_img, query_cmask, support_imgs, support_cmasks, org_qry_imsize = self.load_frame(query_name,
                                                                                             support_names)

        query_mask = self.extract_ignore_idx(query_cmask, class_sample_query)[0]

        if self.image_transform:
            query_img = self.image_transform(query_img)

        if self.mask_transform:
            query_mask = self.mask_transform(query_mask)

        if self.image_transform:
            for i in range(len(support_imgs)):
                support_imgs[i] = self.image_transform(support_imgs[i])
        support_masks = [self.extract_ignore_idx(m, class_sample_support)[0] for m in support_cmasks]
        if self.mask_transform:
            for i in range(len(support_masks)):
                support_masks[i] = self.mask_transform(support_masks[i])

        batch = {'query_img': query_img,
                 'query_mask': query_mask,
                 'query_name': query_name,
                 'org_query_imsize': org_qry_imsize,
                 'support_imgs': support_imgs,
                 'support_masks': support_masks,
                 'class_id': torch.tensor(class_sample_query),
                 }

        return batch

    def extract_ignore_idx(self, mask, class_id):
        mask = np.array(mask)
        boundary = np.floor(mask / 255.)
        mask[mask != class_id + 1] = 0.
        mask[mask == class_id + 1] = 255.
        return Image.fromarray(mask), boundary

    def load_frame(self, query_name, support_names):
        query_img = self.read_img(query_name)
        query_mask = self.read_mask(query_name)
        support_imgs = []
        support_masks = []
        for support_name in support_names:
            support_imgs.append(self.read_img(support_name))
            support_masks.append(self.read_mask(support_name))

        org_qry_imsize = query_img.size

        return query_img, query_mask, support_imgs, support_masks, org_qry_imsize

    def read_mask(self, img_name):
        r"""Return segmentation mask in PIL Image"""
        mask = Image.open(os.path.join(self.ann_path, img_name) + '.png')
        return mask

    def read_img(self, img_name):
        r"""Return RGB image in PIL Image"""
        return Image.open(os.path.join(self.img_path, img_name) + '.jpg')

    def sample_episode(self, idx):
        """Returns the index of the query, support and class."""
        query_name, class_sample = self.img_metadata[idx]
        if not self.random:
            support_class = class_sample
        else:
            support_class = \
            np.random.choice([k for k in self.img_metadata_classwise.keys() if self.img_metadata_classwise[k]], 1,
                             replace=False)[0]

        support_names = []
        while True:  # keep sampling support set if query == support
            support_name = np.random.choice(self.img_metadata_classwise[support_class], 1, replace=False)[0]
            if query_name != support_name and support_name not in support_names:
                support_names.append(support_name)
                if len(support_names) >= self.num_supports:
                    break
        return query_name, support_names, class_sample, support_class

    def build_class_ids(self):
        nclass_trn = self.nclass // self.nfolds
        class_ids_val = [self.fold * nclass_trn + i for i in range(nclass_trn)]
        return class_ids_val

    def build_img_metadata(self):

        def read_metadata(split, fold_id):
            cwd = os.path.dirname(os.path.abspath(__file__))
            fold_n_metadata = os.path.join(cwd, 'splits/pascal/%s/fold%d.txt' % (split, fold_id))
            with open(fold_n_metadata, 'r') as f:
                fold_n_metadata = f.read().split('\n')[:-1]
            fold_n_metadata = [[data.split('__')[0], int(data.split('__')[1]) - 1] for data in fold_n_metadata]
            return fold_n_metadata

        img_metadata = []
        img_metadata = read_metadata('val', self.fold)

        print('Total (val) images are : %d' % len(img_metadata))

        return img_metadata

    def build_img_metadata_classwise(self):
        img_metadata_classwise = {}
        for class_id in range(self.nclass):
            img_metadata_classwise[class_id] = []

        for img_name, img_class in self.img_metadata:
            img_metadata_classwise[img_class] += [img_name]
        return img_metadata_classwise