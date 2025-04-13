import os
import glob
import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageOps


unseen_classes_sketchy = [
    "bat", "cabin", "cow", "dolphin", "door", "giraffe", "helicopter", "mouse", "pear", "raccoon", "rhinoceros",
    "saw", "scissors", "seagull", "skyscraper", "songbird", "sword", "tree", "wheelchair", "windmill", "window",
]
unseen_classes_tuberlin = [
    "ant", "banana", "bottle opener", "brain", "bread", "bridge", "bus", "canoe", "fan", "frying-pan",
    "horse", "hot air balloon", "laptop", "lighter", "parachute", "penguin", "pizza", "rollerblades",
    "shoe", "snowboard", "space shuttle", "streetlight", "suitcase", "table", "teacup", "telephone",
    "t-shirt", "tractor", "trombone", "windmill"
]
unseen_classes_quickdraw = [
    "banana", "bat", "beach", "bread", "cactus", "cake", "campfire", "cow", "dolphin", "door", "fire_hydrant",
    "feather", "fan", "frog", "giraffe", "hamburger", "helicopter", "megaphone", "mouse", "palm tree", "raccoon",
    "rhinoceros", "shark", "saw", "scissors", "skyscraper", "tree", "tiger", "windmill", "zebra"
]

# Path to 'Sketchy' folder holding Sketch_extended dataset. It should have 2 folders named 'sketch' and 'photo'.
class Sketchy(torch.utils.data.Dataset):
    def __init__(self, opts, transform, mode='train', used_cat=None, return_orig=False):
        self.opts = opts
        self.transform = transform
        self.return_orig = return_orig
        self.data_dir = './datasets/Sketchy/'            # Modify
        self.all_categories = os.listdir(os.path.join(self.data_dir, 'sketch'))
        if '.ipynb_checkpoints' in self.all_categories:
            self.all_categories.remove('.ipynb_checkpoints')

        if self.opts.data_split > 0:
            np.random.shuffle(self.all_categories)
            if used_cat is None:
                self.all_categories = self.all_categories[:int(len(self.all_categories) * self.opts.data_split)]
            else:
                self.all_categories = list(set(self.all_categories) - set(used_cat))
        else:
            if mode == 'train':
                self.all_categories = list(set(self.all_categories) - set(unseen_classes_sketchy))
            else:
                self.all_categories = unseen_classes_sketchy

        self.all_sketches_path = []
        self.all_photos_path = {}

        for category in self.all_categories:
            self.all_sketches_path.extend(glob.glob(os.path.join(self.data_dir, 'sketch', category, '*.png')))
            self.all_photos_path[category] = glob.glob(os.path.join(self.data_dir, 'photo', category, '*.jpg'))

    def __len__(self):
        return len(self.all_sketches_path)

    def __getitem__(self, index):
        filepath = self.all_sketches_path[index]
        category = filepath.split(os.path.sep)[-2]  # hamburger(sketch, also for image)
        filename = os.path.basename(filepath)  # n07697313_2730-2.png(sketch)

        neg_classes = self.all_categories.copy()
        neg_classes.remove(category)

        sk_path = filepath
        img_path = np.random.choice(self.all_photos_path[category])
        neg_path = np.random.choice(self.all_photos_path[np.random.choice(neg_classes)])


        sk_data = ImageOps.pad(Image.open(sk_path).convert('RGB'), size=(self.opts.max_size, self.opts.max_size))
        img_data = ImageOps.pad(Image.open(img_path).convert('RGB'), size=(self.opts.max_size, self.opts.max_size))
        neg_data = ImageOps.pad(Image.open(neg_path).convert('RGB'), size=(self.opts.max_size, self.opts.max_size))

        sk_tensor = self.transform(sk_data)
        img_tensor = self.transform(img_data)
        neg_tensor = self.transform(neg_data)

        if self.return_orig:
            return (sk_tensor, img_tensor, neg_tensor, category, filename, sk_data, img_data, neg_data)
        else:
            return (sk_tensor, img_tensor, neg_tensor, category, filename)

    @staticmethod
    def data_transform(opts):
        dataset_transforms = transforms.Compose([
            transforms.Resize((opts.max_size, opts.max_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return dataset_transforms


class TUBerlin(torch.utils.data.Dataset):
    def __init__(self, opts, transform, mode='train', used_cat=None, return_orig=False):
        self.opts = opts
        self.transform = transform
        self.return_orig = return_orig
        self.data_dir = './datasets/TUBerlin/'                   # Modify
        self.all_categories = os.listdir(os.path.join(self.data_dir, 'png_ready'))
        if '.ipynb_checkpoints' in self.all_categories:
            self.all_categories.remove('.ipynb_checkpoints')

        if self.opts.data_split > 0:
            np.random.shuffle(self.all_categories)
            if used_cat is None:
                self.all_categories = self.all_categories[:int(len(self.all_categories) * self.opts.data_split)]
            else:
                self.all_categories = list(set(self.all_categories) - set(used_cat))
        else:
            if mode == 'train':
                self.all_categories = list(set(self.all_categories) - set(unseen_classes_tuberlin))
            else:
                self.all_categories = unseen_classes_tuberlin


        self.all_sketches_path = []
        self.all_photos_path = {}

        for category in self.all_categories:
            self.all_sketches_path.extend(glob.glob(os.path.join(self.data_dir, 'png_ready', category, '*.png')))
            self.all_photos_path[category] = glob.glob(os.path.join(self.data_dir, 'ImageResized_ready', category, '*.*'))

    def __len__(self):
        return len(self.all_sketches_path)

    def __getitem__(self, index):
        filepath = self.all_sketches_path[index]
        category = filepath.split(os.path.sep)[-2]
        filename = os.path.basename(filepath)

        neg_classes = self.all_categories.copy()
        neg_classes.remove(category)

        sk_path = filepath
        img_path = np.random.choice(self.all_photos_path[category])     # 随机选择
        neg_path = np.random.choice(self.all_photos_path[np.random.choice(neg_classes)])    # 随机选择


        sk_data = ImageOps.pad(Image.open(sk_path).convert('RGB'), size=(self.opts.max_size, self.opts.max_size))

        img_data = ImageOps.pad(Image.open(img_path).convert('RGB'), size=(self.opts.max_size, self.opts.max_size))

        neg_data = ImageOps.pad(Image.open(neg_path).convert('RGB'), size=(self.opts.max_size, self.opts.max_size))

        sk_tensor = self.transform(sk_data)
        img_tensor = self.transform(img_data)
        neg_tensor = self.transform(neg_data)

        if self.return_orig:
            return (sk_tensor, img_tensor, neg_tensor, category, filename,
                    sk_data, img_data, neg_data)
        else:
            return (sk_tensor, img_tensor, neg_tensor, category, filename)

    @staticmethod
    def data_transform(opts):
        dataset_transforms = transforms.Compose([
            transforms.Resize((opts.max_size, opts.max_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return dataset_transforms

class QuickDraw(torch.utils.data.Dataset):
    def __init__(self, opts, transform, mode='train', used_cat=None, return_orig=False):
        self.opts = opts
        self.transform = transform
        self.return_orig = return_orig
        self.data_dir = './datasets/QuickDraw/'              # Modify

        self.all_categories = os.listdir(os.path.join(self.data_dir, 'sketches'))
        if '.ipynb_checkpoints' in self.all_categories:
            self.all_categories.remove('.ipynb_checkpoints')

        if self.opts.data_split > 0:
            np.random.shuffle(self.all_categories)
            if used_cat is None:
                self.all_categories = self.all_categories[:int(len(self.all_categories) * self.opts.data_split)]
            else:
                self.all_categories = list(set(self.all_categories) - set(used_cat))
        else:
            if mode == 'train':
                self.all_categories = list(set(self.all_categories) - set(unseen_classes_quickdraw))
            else:
                self.all_categories = unseen_classes_quickdraw

        self.all_sketches_path = []
        self.all_photos_path = {}

        for category in self.all_categories:
            self.all_sketches_path.extend(glob.glob(os.path.join(self.data_dir, 'sketches', category, '*.png')))
            self.all_photos_path[category] = glob.glob(os.path.join(self.data_dir, 'images', category, '*.*'))

    def __len__(self):
        return len(self.all_sketches_path)

    def __getitem__(self, index):
        filepath = self.all_sketches_path[index]
        category = filepath.split(os.path.sep)[-2]
        filename = os.path.basename(filepath)

        neg_classes = self.all_categories.copy()
        neg_classes.remove(category)

        sk_path = filepath
        img_path = np.random.choice(self.all_photos_path[category])

        neg_path = np.random.choice(self.all_photos_path[np.random.choice(neg_classes)])

        sk_data = ImageOps.pad(Image.open(sk_path).convert('RGB'), size=(self.opts.max_size, self.opts.max_size))

        img_data = ImageOps.pad(Image.open(img_path).convert('RGB'), size=(self.opts.max_size, self.opts.max_size))

        neg_data = ImageOps.pad(Image.open(neg_path).convert('RGB'), size=(self.opts.max_size, self.opts.max_size))


        sk_tensor = self.transform(sk_data)
        img_tensor = self.transform(img_data)
        neg_tensor = self.transform(neg_data)

        if self.return_orig:
            return (sk_tensor, img_tensor, neg_tensor, category, filename,
                    sk_data, img_data, neg_data)
        else:
            return (sk_tensor, img_tensor, neg_tensor, category, filename)

    @staticmethod
    def data_transform(opts):
        dataset_transforms = transforms.Compose([
            transforms.Resize((opts.max_size, opts.max_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return dataset_transforms

if __name__ == '__main__':
    from utils.options import opts
    import tqdm

    dataset_transforms = Sketchy.data_transform(opts)
    dataset_train = Sketchy(opts, dataset_transforms, mode='train', return_orig=True)
    dataset_val = Sketchy(opts, dataset_transforms, mode='val', used_cat=dataset_train.all_categories, return_orig=True)

    idx = 0
    for data in tqdm.tqdm(dataset_val):
        continue
        (sk_tensor, img_tensor, neg_tensor, filename,
         sk_data, img_data, neg_data) = data

        canvas = Image.new('RGB', (224 * 3, 224))
        offset = 0
        for im in [sk_data, img_data, neg_data]:
            canvas.paste(im, (offset, 0))
            offset += im.size[0]
        canvas.save('output/%d.jpg' % idx)
        idx += 1
