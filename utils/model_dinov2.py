import numpy as np
from torchmetrics.functional import retrieval_average_precision
import pytorch_lightning as pl
from utils.options import opts
from model import *


def freeze_model(m):
    m.requires_grad_(False)


def distance_fn_(x, y):
    return 1.0 - F.cosine_similarity(x, y, dim=-1)


class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.opts = opts
        self.model_dino = vit_base(img_size=518, patch_size=14, init_values=1.0, ffn_layer="mlp", block_chunks=0)
        path = "./utils/dinov2_vitb14_pretrain.pth"                    # Modify
        checkpoint = torch.load(path, map_location='cpu')
        info = self.model_dino.load_state_dict(checkpoint, strict=False)

        print("info.missing_keys: ", info.missing_keys)
        print("info.unexpected_keys: ", info.unexpected_keys)
        # print(self.model_dino.state_dict())

        # Sketchy
        self.loss_fn = nn.TripletMarginWithDistanceLoss(distance_function=distance_fn_, margin=0.2)

        # TU-Berlin、QuickDraw
        # self.loss_fn = nn.TripletMarginWithDistanceLoss(distance_function=distance_fn_, margin=0.04)

        self.best_metric = -1e3

    def configure_optimizers(self):
        print("here attention! We freeze partial params!")
        params_to_update = [param for param in self.model_dino.parameters() if param.requires_grad]
        print("len(params_to_update): ", len(params_to_update))
        optimizer = torch.optim.Adam([{'params': params_to_update, 'lr': self.opts.clip_LN_lr}])

        return optimizer

    def forward(self, data, dtype='image'):
        if dtype == 'image':
            feat = self.model_dino(data, dtype)
        else:
            feat = self.model_dino(data, dtype)

        return feat

    def training_step(self, batch, batch_idx):
        sk_tensor, img_tensor, neg_tensor, category = batch[:4]

        img_feat = self.forward(img_tensor, dtype='image')
        sk_feat = self.forward(sk_tensor, dtype='sketch')
        neg_feat = self.forward(neg_tensor, dtype='image')

        loss = self.loss_fn(sk_feat, img_feat, neg_feat)

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        sk_tensor, img_tensor, neg_tensor, category = batch[:4]

        img_feat = self.forward(img_tensor, dtype='image')
        sk_feat = self.forward(sk_tensor, dtype='sketch')
        neg_feat = self.forward(neg_tensor, dtype='image')

        loss = self.loss_fn(sk_feat, img_feat, neg_feat)
        self.log('val_loss', loss)
        return sk_feat, img_feat, category

    def validation_epoch_end(self, val_step_outputs):
        Len = len(val_step_outputs)

        if Len == 0:
            return

        query_feat_all = torch.cat([val_step_outputs[i][0] for i in range(Len)])
        gallery_feat_all = torch.cat([val_step_outputs[i][1] for i in range(Len)])
        all_category = np.array(sum([list(val_step_outputs[i][2]) for i in range(Len)], []))

        # mAP category-level SBIR Metrics
        gallery = gallery_feat_all
        ap = torch.zeros(len(query_feat_all))
        for idx, sk_feat in enumerate(query_feat_all):
            category = all_category[idx]

            distance = -1 * distance_fn_(sk_feat.unsqueeze(0), gallery)

            target = torch.zeros(len(gallery), dtype=torch.bool)
            target[np.where(all_category == category)] = True
            ap[idx] = retrieval_average_precision(distance.cpu(), target.cpu())

        mAP = torch.mean(ap)
        self.log('mAP', mAP)
        if self.global_step > 0:
            self.best_metric = self.best_metric if (self.best_metric > mAP.item()) else mAP.item()

        print('mAP: {}, Best mAP: {}'.format(mAP.item(), self.best_metric))
