
import os
import random
import shutil
import sys
from datetime import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import gdown


class Logger(object):
    """Reference: https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514"""

    def __init__(self, fn, eval, ask=True):
        if not os.path.exists("./results/"):
            os.mkdir("./results/")

        logdir = self._make_dir(fn)
        self.eval = eval
        
        if not os.path.exists(logdir):
            os.mkdir(logdir)

        # if len(os.listdir(logdir)) != 0 and ask:
        #     exit(1)

        self.set_dir(logdir)

    def _make_dir(self, fn):
        # today = datetime.today().strftime("%y%m%d")
        logdir = f'./results/{fn}/'
        return logdir

    def set_dir(self, logdir, log_fn='log.txt'):
        self.logdir = logdir
        if not os.path.exists(logdir):
            os.mkdir(logdir)

        if self.eval:
            log_fn = 'eval_' + log_fn
        else:
            self.writer = SummaryWriter(logdir)
        self.log_file = open(os.path.join(logdir, log_fn), 'a')

    def log(self, string):
        self.log_file.write('[%s] %s' % (datetime.now(), string) + '\n')
        self.log_file.flush()

        print('[%s] %s' % (datetime.now(), string))
        sys.stdout.flush()

    def log_dirname(self, string):
        self.log_file.write('%s (%s)' % (string, self.logdir) + '\n')
        self.log_file.flush()

        print('%s (%s)' % (string, self.logdir))
        sys.stdout.flush()

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        self.writer.add_scalar(tag, value, step)

    def image_summary(self, tag, images, step):
        """Log a list of images."""
        self.writer.add_image(tag, images, step)

    def video_summary(self, tag, videos, step):
        self.writer.add_video(tag, videos, step, fps=16)

    def histo_summary(self, tag, values, step):
        """Log a histogram of the tensor of values."""
        self.writer.add_histogram(tag, values, step, bins='auto')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.value = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.value = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.average = self.sum / self.count


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def file_name(args):
    fn = f'{args.exp}_{args.id}_{args.data}'
    fn += f'_{args.seed}'
    return fn


def psnr(mse):
    """
    Computes PSNR from MSE.
    """
    return -10.0 * mse.log10()

def download(id, fname, root=os.path.expanduser('~/.cache/video-diffusion')):
    os.makedirs(root, exist_ok=True)
    destination = os.path.join(root, fname)

    if os.path.exists(destination):
        return destination

    gdown.download(id=id, output=destination, quiet=False)
    return destination


def make_pairs(l, t1, t2, num_pairs, given_vid):
    B, T, C, H, W = given_vid.size()
    idx1 = t1.view(B, num_pairs, 1, 1, 1, 1).expand(B, num_pairs, 1, C, H, W).type(torch.int64)
    frame1 = torch.gather(given_vid.unsqueeze(1).repeat(1,num_pairs, 1,1,1,1), 2, idx1).squeeze()
    t1 = t1.float() / (l - 1) 

    idx2 = t2.view(B, num_pairs, 1, 1, 1, 1).expand(B, num_pairs, 1, C, H, W).type(torch.int64)
    frame2 = torch.gather(given_vid.unsqueeze(1).repeat(1,num_pairs,1,1,1,1), 2, idx2).squeeze()
    t2 = t2.float() / (l - 1) 

    frame1 = frame1.view(-1, C, H, W)
    frame2 = frame2.view(-1, C, H, W)

    # sort by t
    t1 = t1.view(-1, 1, 1, 1).repeat(1, C, H, W)
    t2 = t2.view(-1, 1, 1, 1).repeat(1, C, H, W)

    ret_frame1 = torch.where(t1 < t2, frame1, frame2)
    ret_frame2 = torch.where(t1 < t2, frame2 ,frame1)

    t1 = t1[:, 0:1]
    t2 = t2[:, 0:1]

    ret_t1 = torch.where(t1 < t2, t1, t2)
    ret_t2 = torch.where(t1 < t2, t2, t1)

    dt = ret_t2 - ret_t1

    return torch.cat([ret_frame1, ret_frame2, dt], dim=1)

def make_mixed_pairs(l, t1, t2, given_vid_real, given_vid_fake):
    B, T, C, H, W = given_vid_real.size()
    idx1 = t1.view(-1, 1, 1, 1, 1).expand(B, 1, C, H, W).type(torch.int64)
    frame1 = torch.gather(given_vid_real, 1, idx1).squeeze()
    t1 = t1.float() / (l - 1) 

    idx2 = t2.view(-1, 1, 1, 1, 1).expand(B, 1, C, H, W).type(torch.int64)
    frame2 = torch.gather(given_vid_fake, 1, idx2).squeeze()
    t2 = t2.float() / (l - 1) 


    # sort by t
    t1 = t1.view(-1, 1, 1, 1).repeat(1, C, H, W)
    t2 = t2.view(-1, 1, 1, 1).repeat(1, C, H, W)

    ret_frame1 = torch.where(t1 < t2, frame1, frame2)
    ret_frame2 = torch.where(t1 < t2, frame2 ,frame1)

    t1 = t1[:, 0:1]
    t2 = t2[:, 0:1]

    ret_t1 = torch.where(t1 < t2, t1, t2)
    ret_t2 = torch.where(t1 < t2, t2, t1)

    dt = ret_t2 - ret_t1

    return torch.cat([ret_frame1, ret_frame2, dt], dim=1)

