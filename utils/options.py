import argparse

parser = argparse.ArgumentParser(description='Sketch-based image retrieval')

parser.add_argument('--exp_name', type=str, default='CMCA')
parser.add_argument('--max_size', type=int, default=224)
parser.add_argument('--data_split', type=float, default=-1.0)


# -----------------
# Training Params
# ----------------------
parser.add_argument('--clip_lr', type=float, default=1e-4)
parser.add_argument('--clip_LN_lr', type=float, default=1e-6)
parser.add_argument('--prompt_lr', type=float, default=1e-4)
parser.add_argument('--linear_lr', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=30)    # dinov2_model(two gpus)
parser.add_argument('--workers', type=int, default=16)

opts = parser.parse_args()
