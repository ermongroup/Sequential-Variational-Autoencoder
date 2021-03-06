

# --dataset=celebA --denoise_train --plot_reconstruction --gpus=0 --db_path=/ssd_data/CelebA --use_gui
# --dataset=lsun --denoise_train --plot_reconstruction --gpus=1 --db_path=/data/data/lsun/bedroom --use_gui

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpus', type=str, default='')
parser.add_argument('--dataset', type=str, default='celebA')
parser.add_argument('--netname', type=str, default='')
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--db_path', type=str, default='')
parser.add_argument('--denoise_train', dest='denoise_train', action='store_true',
                    help='Use denoise training by adding Gaussian/salt and pepper noise')
parser.add_argument('--plot_reconstruction', dest='plot_reconstruction', action='store_true',
                    help='Plot reconstruction')
parser.add_argument('--use_gui', dest='use_gui', action='store_true',
                    help='Display the results with a GUI window')
parser.add_argument('--vis_frequency', type=int, default=1000,
                    help='How many train batches before we perform visualization')
args = parser.parse_args()

import matplotlib
if not args.use_gui:
    matplotlib.use('Agg')
else:
    from matplotlib import pyplot as plt
    plt.ion()
    plt.show()

from dataset import *
from sequential_vae import SequentialVAE
from trainer import NoisyTrainer

if args.gpus is not '':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

if args.dataset == 'mnist':
    dataset = MnistDataset()
elif args.dataset == 'lsun':
    dataset = LSUNDataset(db_path=args.db_path)
elif args.dataset == 'celebA':
    dataset = CelebADataset(db_path=args.db_path)
elif args.dataset == 'svhn':
    dataset = SVHNDataset(db_path=args.db_path)
else:
    print("Unknown dataset")
    exit(-1)

model = SequentialVAE(dataset, name=args.netname, batch_size=args.batch_size)
trainer = NoisyTrainer(model, dataset, args)
trainer.train()

