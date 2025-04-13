import os
import numpy as np
from torch.utils.data import Dataset
from skimage.transform import warp, AffineTransform
from nltk.corpus import wordnet as wn
import pickle
from PIL import Image, ImageOps

class SketchyDataset(Dataset):
    def __init__(self, split='train',
                 root_dir='./datasets/Sketchy',          # Modify
                 version='sketch_tx_000000000000_ready', zero_version='zeroshot2',
                 cid_mask=False, transform=None, aug=False, shuffle=False, first_n_debug=9999999):

        self.root_dir = root_dir
        self.version = version
        self.split = split

        self.img_dir = self.root_dir + '/zeroshot2'

        if self.split == 'train':
            file_ls_file = os.path.join(self.root_dir, zero_version, self.version + '_filelist_train.txt')
        elif self.split == 'val':
            file_ls_file = os.path.join(self.root_dir, zero_version, self.version + '_filelist_test.txt')
        elif self.split == 'zero':
            file_ls_file = os.path.join(self.root_dir, zero_version, self.version + '_filelist_zero.txt')
        else:
            print('unknown split for dataset initialization: ' + self.split)
            return

        with open(file_ls_file, 'r') as fh:
            file_content = fh.readlines()

        self.file_ls = np.array([' '.join(ff.strip().split()[:-1]) for ff in file_content])
        self.labels = np.array([int(ff.strip().split()[-1]) for ff in file_content])
        if shuffle:
            self.shuffle()

        self.file_ls = self.file_ls[:first_n_debug]
        self.labels = self.labels[:first_n_debug]

        self.transform = transform
        self.aug = aug

        self.cid_mask = cid_mask
        if cid_mask:
            cid_mask_file = os.path.join(self.root_dir, zero_version, 'cid_mask.pickle')
            with open(cid_mask_file, 'rb') as fh:
                self.cid_matrix = pickle.load(fh)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        filepath = os.path.join(self.img_dir, self.file_ls[idx])

        img = ImageOps.pad(Image.open(filepath).convert('RGB'), size=(224, 224))

        tensor = self.transform(img)
        label = self.labels[idx]

        return tensor, label

    def shuffle(self):
        s_idx = np.random.shuffle(np.arange(len(self.labels)))
        self.file_ls = self.file_ls[s_idx]
        self.labels = self.labels[s_idx]


class TUBerlinDataset(Dataset):
    def __init__(self, split='train',
                 root_dir='./datasets/TUBerlin',                 # Modify
                 version='png_ready', zero_version='zeroshot', \
                 cid_mask=False, transform=None, aug=False, shuffle=False, first_n_debug=9999999):

        self.root_dir = root_dir
        self.version = version
        self.split = split

        self.img_dir = self.root_dir

        if self.split == 'train':
            file_ls_file = os.path.join(self.root_dir, zero_version, self.version + '_filelist_train.txt')
        elif self.split == 'val':
            file_ls_file = os.path.join(self.root_dir, zero_version, self.version + '_filelist_test.txt')
        elif self.split == 'zero':
            file_ls_file = os.path.join(self.root_dir, zero_version, self.version + '_filelist_zero.txt')
        else:
            print('unknown split for dataset initialization: ' + self.split)
            return

        with open(file_ls_file, 'r') as fh:
            file_content = fh.readlines()

        self.file_ls = np.array([' '.join(ff.strip().split()[:-1]) for ff in file_content])
        self.labels = np.array([int(ff.strip().split()[-1]) for ff in file_content])
        if shuffle:
            self.shuffle()

        self.file_ls = self.file_ls[:first_n_debug]
        self.labels = self.labels[:first_n_debug]

        self.transform = transform
        self.aug = aug

        self.cid_mask = cid_mask
        if cid_mask:
            cid_mask_file = os.path.join(self.root_dir, zero_version, 'cid_mask.pickle')
            with open(cid_mask_file, 'rb') as fh:
                self.cid_matrix = pickle.load(fh)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        filepath = os.path.join(self.img_dir, self.file_ls[idx])

        img = ImageOps.pad(Image.open(filepath).convert('RGB'), size=(224, 224))

        tensor = self.transform(img)
        label = self.labels[idx]

        return tensor, label

    def shuffle(self):
        s_idx = np.random.shuffle(np.arange(len(self.labels)))
        self.file_ls = self.file_ls[s_idx]
        self.labels = self.labels[s_idx]


class QuickDrawDataset(Dataset):
    def __init__(self, split='train',
                 root_dir='./datasets/QuickDraw',                # Modify
                 version='sketch', zero_version='zeroshot',
                 cid_mask=False, transform=None, aug=False, shuffle=False, first_n_debug=9999999):

        self.root_dir = root_dir
        self.version = version
        self.split = split

        self.img_dir = self.root_dir

        if self.split == 'train':
            file_ls_file = os.path.join(self.root_dir, zero_version, self.version + '_filelist_train.txt')
        elif self.split == 'val':
            file_ls_file = os.path.join(self.root_dir, zero_version, self.version + '_filelist_test.txt')
        elif self.split == 'zero':
            file_ls_file = os.path.join(self.root_dir, zero_version, self.version + '_filelist_zero.txt')
        else:
            print('unknown split for dataset initialization: ' + self.split)
            return

        with open(file_ls_file, 'r') as fh:
            file_content = fh.readlines()

        self.file_ls = np.array([' '.join(ff.strip().split()[:-1]) for ff in file_content])
        self.labels = np.array([int(ff.strip().split()[-1]) for ff in file_content])
        if shuffle:
            self.shuffle()

        self.file_ls = self.file_ls[:first_n_debug]
        self.labels = self.labels[:first_n_debug]

        self.transform = transform
        self.aug = aug

        self.cid_mask = cid_mask
        if cid_mask:
            cid_mask_file = os.path.join(self.root_dir, zero_version, 'cid_mask.pickle')
            with open(cid_mask_file, 'rb') as fh:
                self.cid_matrix = pickle.load(fh)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        filepath = os.path.join(self.img_dir, self.file_ls[idx])

        img = ImageOps.pad(Image.open(filepath).convert('RGB'), size=(224, 224))

        tensor = self.transform(img)
        label = self.labels[idx]

        return tensor, label

    def shuffle(self):
        s_idx = np.random.shuffle(np.arange(len(self.labels)))
        self.file_ls = self.file_ls[s_idx]
        self.labels = self.labels[s_idx]

