import os
import sys
import torch
import argparse
import cv2
from skimage import morphology
import numpy as np
import torch.nn as nn
import scipy.io as sio
import cv2
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader
from os.path import join, isdir, abspath, dirname

# Customized import.
from networks import HED
from datasets import MyDataset
from utils import Logger, load_vgg16_caffe, load_pretrained_caffe


# Parse arguments.
parser = argparse.ArgumentParser(description='HED training.')
# 1. Actions.
parser.add_argument('--test',             default=False,             help='Only test the model.', action='store_true')
# 2. Counts.
parser.add_argument('--train_batch_size', default=1,    type=int,   metavar='N', help='Training batch size.')
parser.add_argument('--test_batch_size',  default=1,    type=int,   metavar='N', help='Test batch size.')
parser.add_argument('--train_iter_size',  default=10,   type=int,   metavar='N', help='Training iteration size.')
parser.add_argument('--max_epoch',        default=40,   type=int,   metavar='N', help='Total epochs.')
parser.add_argument('--print_freq',       default=500,  type=int,   metavar='N', help='Print frequency.')
# 3. Optimizer settings.
parser.add_argument('--lr',               default=1e-6, type=float, metavar='F', help='Initial learning rate.')
parser.add_argument('--lr_stepsize',      default=1e4,  type=int,   metavar='N', help='Learning rate step size.')
# Note: Step size is based on number of iterations, not number of batches.
#   https://github.com/s9xie/hed/blob/94fb22f10cbfec8d84fbc0642b224022014b6bd6/src/caffe/solver.cpp#L498
parser.add_argument('--lr_gamma',         default=0.1,  type=float, metavar='F', help='Learning rate decay (gamma).')
parser.add_argument('--momentum',         default=0.9,  type=float, metavar='F', help='Momentum.')
parser.add_argument('--weight_decay',     default=2e-4, type=float, metavar='F', help='Weight decay.')
# 4. Files and folders.
parser.add_argument('--vgg16_caffe',      default='',                help='Resume VGG-16 Caffe parameters.')
parser.add_argument('--checkpoint',       default='',                help='Resume the checkpoint.')
parser.add_argument('--caffe_model',      default='',                help='Resume HED Caffe model.')
parser.add_argument('--output',           default='./output',        help='Output folder.')
parser.add_argument('--dataset',          default='./data/HED-BSDS', help='HED-BSDS dataset folder.')
# 5. Others.
parser.add_argument('--cpu',              default=False,             help='Enable CPU mode.', action='store_true')
args = parser.parse_args()

# 输入的参数
args.caffe_model = './data/hed_pretrained_bsds.py36pickle'
args.test = True
args.dataset = './testimg'

# Set device.
device = torch.device('cpu' if args.cpu else 'cuda')


def main():
    ################################################
    # I. Miscellaneous.
    ################################################
    # Create the output directory.
    current_dir = abspath(dirname(__file__))
    output_dir = join(current_dir, args.output)
    if not isdir(output_dir):
        os.makedirs(output_dir)

    # Set logger.
    now_str = datetime.now().strftime('%y%m%d-%H%M%S')
    log = Logger(join(output_dir, 'log-{}.txt'.format(now_str)))
    sys.stdout = log  # Overwrite the standard output.

    ################################################
    # II. Datasets.
    ################################################
    # Datasets and dataloaders.
    test_dataset = MyDataset(dataset_dir=args.dataset)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size,
                             num_workers=4, drop_last=False, shuffle=False)

    ################################################
    # III. Network and optimizer.
    ################################################
    # Create the network in GPU.
    net = nn.DataParallel(HED(device))
    net.to(device)

    ################################################
    # IV. Pre-trained parameters.
    ################################################
    # Load parameters from pre-trained VGG-16 Caffe model.
    if args.vgg16_caffe:
        load_vgg16_caffe(net, args.vgg16_caffe)

    # Resume the HED Caffe model.
    if args.caffe_model:
        load_pretrained_caffe(net, args.caffe_model)

    test(test_loader, net, save_dir=join(output_dir, 'test'))


def test(test_loader, net, save_dir):
    """ Test procedure. """
    # Create the directories.
    if not isdir(save_dir):
        os.makedirs(save_dir)
    save_png_dir = join(save_dir, 'png')
    if not isdir(save_png_dir):
        os.makedirs(save_png_dir)
    save_mat_dir = join(save_dir, 'mat')
    if not isdir(save_mat_dir):
        os.makedirs(save_mat_dir)

    # Switch to evaluation mode.
    net.eval()
    # Generate predictions and save.
    assert args.test_batch_size == 1  # Currently only support test batch size 1.
    for batch_index, images in enumerate(tqdm(test_loader)):
        images = images.cuda()
        _, _, h, w = images.shape
        preds_list = net(images)
        fuse       = preds_list[-1].detach().cpu().numpy()[0, 0]  # Shape: [h, w].
        name       = test_loader.dataset.images_name[batch_index]

        edgeImage = Image.fromarray((fuse * 255).astype(np.uint8))

        # 转为二值图;
        grayEdgeImg = cv2.cvtColor(np.asarray(edgeImage), cv2.COLOR_RGB2BGR)
        grayEdgeImg = cv2.cvtColor(grayEdgeImg, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(grayEdgeImg, 128, 255, cv2.THRESH_BINARY)

        #提取骨架
        binary[binary==255] = 1
        skeleton0 = morphology.skeletonize(binary)
        skeleton = skeleton0.astype(np.uint8) * 255
        cv2.imwrite(join(save_png_dir, '{}_skelton.jpg'.format(name)), skeleton)

        # 重新保存
        binary[binary == 1] = 255
        finalEdgeImage = Image.fromarray(cv2.cvtColor(binary, cv2.COLOR_BGR2RGB))
        finalEdgeImage.save(join(save_png_dir, '{}_edge.png'.format(name)))
        # print('Test batch {}/{}.'.format(batch_index + 1, len(test_loader)))


if __name__ == '__main__':
    main()
