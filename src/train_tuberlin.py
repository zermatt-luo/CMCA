import os
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from utils.model_dinov2 import Model
from utils.dataset_train import TUBerlin
from utils.options import opts

if __name__ == '__main__':
    dataset_transforms = TUBerlin.data_transform(opts)

    train_dataset = TUBerlin(opts, dataset_transforms, mode='train', return_orig=False)
    val_dataset = TUBerlin(opts, dataset_transforms, mode='val', used_cat=train_dataset.all_categories,
                           return_orig=False)

    train_loader = DataLoader(dataset=train_dataset, batch_size=opts.batch_size, num_workers=opts.workers)
    val_loader = DataLoader(dataset=val_dataset, batch_size=opts.batch_size, num_workers=opts.workers)

    logger = TensorBoardLogger('./checkpoints/tb_logs', name='TU-Berlin/')
    dirpath = './checkpoints/saved_models/%s' % 'TU-Berlin/'
    print(dirpath)

    # checkpoint_callback = ModelCheckpoint(
    #     dirpath=dirpath,
    #     filename="{epoch:02d}-{top10:.2f}",
    #     save_last=True,
    #     save_top_k=-1,
    #     every_n_epochs=1
    # )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=dirpath,
        filename="{epoch:02d}-{top10:.2f}",
        mode='min',
        save_last=True)

    ckpt_path = os.path.join('./checkpoints/saved_models', 'TU-Berlin/', 'last.ckpt')
    if not os.path.exists(ckpt_path):
        ckpt_path = None
    else:
        print('resuming training from %s' % ckpt_path)
    print("ckpt_path: ", ckpt_path)
    trainer = Trainer(
        accelerator='gpu', devices=-1, strategy="ddp_find_unused_parameters_false",     # We use 2 gpus
        min_epochs=1, max_epochs=65,
        benchmark=True,
        logger=logger,
        check_val_every_n_epoch=1,
        resume_from_checkpoint=ckpt_path,
        callbacks=[checkpoint_callback]
    )

dinov2_vitb14 = Model()
# print(dinov2_vitb14)

# Frozen other parameters
for name, param in dinov2_vitb14.named_parameters():
    if 'w_a_' not in name and 'w_b_' not in name:
        param.requires_grad = False

for name, child in dinov2_vitb14.named_parameters():
    print(name, child.requires_grad)

if ckpt_path is None:
    model = dinov2_vitb14
else:
    print('resuming training from %s' % ckpt_path)
    model = dinov2_vitb14.load_from_checkpoint(ckpt_path)
print('beginning training...good luck...')

print(model)

trainer.fit(model, train_loader, val_loader)
