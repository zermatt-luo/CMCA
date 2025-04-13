import warnings

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utils.model_dinov2 import Model
from utils.dataset_evaluation import SketchyDataset
from utils.valid import eval_precision, eval_AP_inner
from utils.options import opts


def distance_fn_(x, y):
    return 1.0 - F.cosine_similarity(x, y, dim=-1)


def get_features(data_loader, dtype='image'):
    model.eval()
    features_all = []
    targets_all = []
    for i, (input, target) in enumerate(data_loader):
        # if i % 10 == 0:
        #     print(i, end=' ', flush=True)
        input = torch.autograd.Variable(input, requires_grad=False).cuda()

        with torch.no_grad():
            if dtype == 'image':
                features = model.model_dino(input, data_type='image')
            else:
                features = model.model_dino(input, data_type='sketch')

        features = F.normalize(features)
        features = features.cpu().detach().numpy()

        features_all.append(features.reshape(input.size()[0], -1))
        targets_all.append(target.detach().numpy())

    features_all = np.concatenate(features_all)
    targets_all = np.concatenate(targets_all)

    return features_all, targets_all


warnings.filterwarnings('ignore')

ckpt_path = './checkpoints/saved_models/Sketchy/epoch=06-top10=0.00.ckpt'

model = Model().load_from_checkpoint(ckpt_path, strict=True)
model = model.cuda()

transformations = transforms.Compose([
    transforms.Resize([opts.max_size, opts.max_size]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

sketchy_zero_ext = SketchyDataset(split='zero', version='all_photo', zero_version='zeroshot2',
                                  transform=transformations, aug=False)

zero_loader_ext = DataLoader(dataset=sketchy_zero_ext, batch_size=80, shuffle=False, num_workers=0)

sketchy_zero = SketchyDataset(split='zero', zero_version='zeroshot2', transform=transformations, aug=False)
zero_loader = DataLoader(dataset=sketchy_zero, batch_size=80, shuffle=False, num_workers=0)


predicted_features_gallery, gt_labels_gallery = get_features(zero_loader_ext, dtype='image')
predicted_features_query, gt_labels_query = get_features(zero_loader, dtype='sketch')


scores_1 = np.zeros((predicted_features_query.shape[0], predicted_features_gallery.shape[0]))

predicted_features_query = torch.tensor(predicted_features_query)
predicted_features_gallery = torch.tensor(predicted_features_gallery)
for idx, sk_feat in enumerate(predicted_features_query):
    distance = -1 * distance_fn_(sk_feat.unsqueeze(0), predicted_features_gallery)  # torch.Size([12553])
    scores_1[idx] = distance


s1 = 0
mAP_ls = [[] for _ in range(len(np.unique(gt_labels_query)))]
for fi in range(predicted_features_query.shape[0]):
    mapi = eval_AP_inner(gt_labels_query[fi], scores_1[fi], gt_labels_gallery, top=200)
    mAP_ls[gt_labels_query[fi]].append(mapi)

for mAPi, mAPs in enumerate(mAP_ls):
    # print(str(mAPi) + ' ' + str(np.nanmean(mAPs)) + ' ' + str(np.nanstd(mAPs)))
    s1 += np.nanmean(mAPs)

mAP = np.array([np.nanmean(maps) for maps in mAP_ls]).mean()

q1 = 0
prec_ls = [[] for _ in range(len(np.unique(gt_labels_query)))]
for fi in range(predicted_features_query.shape[0]):
    prec = eval_precision(gt_labels_query[fi], scores_1[fi], gt_labels_gallery, top=200)
    prec_ls[gt_labels_query[fi]].append(prec)

for preci, precs in enumerate(prec_ls):
    # print(str(preci) + ' ' + str(np.nanmean(precs)) + ' ' + str(np.nanstd(precs)))
    q1 += np.nanmean(precs)
prec = np.array([np.nanmean(pre) for pre in prec_ls]).mean()

print(mAP, prec)
