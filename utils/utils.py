import torch
import torchvision
import os
from torch import distributed, nn
import torchvision.utils as vutils
import random
import numpy as np
from PIL import Image, ImageFont, ImageDraw

def load_model_pytorch(model, load_model, gpu_n=0):
    print("=> loading checkpoint '{}'".format(load_model))

    checkpoint = torch.load(load_model, map_location = lambda storage, loc: storage.cuda(gpu_n))

    if 'state_dict' in checkpoint.keys():
        load_from = checkpoint['state_dict']
    else:
        load_from = checkpoint

    if 1:
        if 'module.' in list(model.state_dict().keys())[0]:
            if 'module.' not in list(load_from.keys())[0]:
                from collections import OrderedDict

                load_from = OrderedDict([("module.{}".format(k), v) for k, v in load_from.items()])

        if 'module.' not in list(model.state_dict().keys())[0]:
            if 'module.' in list(load_from.keys())[0]:
                from collections import OrderedDict

                load_from = OrderedDict([(k.replace("module.", ""), v) for k, v in load_from.items()])

    if 1:
        if list(load_from.items())[0][0][:2] == "1." and list(model.state_dict().items())[0][0][:2] != "1.":
            load_from = OrderedDict([(k[2:], v) for k, v in load_from.items()])

        load_from = OrderedDict([(k, v) for k, v in load_from.items() if "gate" not in k])

    model.load_state_dict(load_from, strict=True)

    epoch_from = -1
    if 'epoch' in checkpoint.keys():
        epoch_from = checkpoint['epoch']
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(load_model, epoch_from))


def create_folder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


random.seed(0)

def distributed_is_initialized():
    if distributed.is_available():
        if distributed.is_initialized():
            return True
    return False


def lr_policy(lr_fn):
    def _alr(optimizer, iteration, epoch):
        lr = lr_fn(iteration, epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return _alr


def lr_cosine_policy(base_lr, warmup_length, epochs):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        return max(5e-3,lr)

    return lr_policy(_lr_fn)

def get_mean_std(use_fp16=False):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    if use_fp16:
        mean = mean.astype('float16')
        std = std.astype('float16')
    return mean, std


def clip(vid, use_fp16=False):
    mean,std = get_mean_std(use_fp16)
    for c in range(3):
        m, s = mean[c], std[c]
        vid[..., c] = torch.clamp(vid[..., c], -m / s, (1 - m) / s)
    return vid


def denormalize(vid, use_fp16=False):
    mean,std = get_mean_std(use_fp16)
    for c in range(3):
        m, s = mean[c], std[c]
        vid[..., c] = torch.clamp(vid[..., c] * s + m, 0, 1)
    return vid


def load_video(video_filepath, hide_portion, num_frames=4, num_masked_frames=1):
    assert hide_portion in ['begin', 'middle', 'end'], '`hide_portion` parameter MUST be one of [`begin`, `middle` , `end`]'
    video = torchvision.io.VideoReader(video_filepath, 'video')
    video.set_current_stream("video")
    frames = [] 
    for frame in video:
        frames.append(frame['data'])



def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def fontsize_dicovery(labels,frame_size):
    # create empty grid in PIL ( be used for overlaying the labels)
    # fond should be adjust according to the lengthiest label
    max_size = 0 
    max_text = ''
    for t in labels: # lengthiest fond discovery happens here
        if len(t) > max_size:
            max_size = len(t)
            max_text = t
            fontsize = 1
            font = ImageFont.truetype("DejaVuSans.ttf", fontsize)
            while font.getsize(max_text)[0] < 0.75*frame_size: # font should only cover 3/4 of the top of the image
                # iterate until the text size is just larger than the criteria
                fontsize += 1
                font = ImageFont.truetype("DejaVuSans.ttf", fontsize)
                fontsize -= 1 # decrease to ensure that it fits the criteria
    return font

def frames_grid(inputs):
    vclip = []
    for t in range(inputs.shape[2]):
        batched_grid = vutils.make_grid(denormalize(inputs[:,:,t,:,:].cpu()),normalize=False, scale_each=True, nrow=int(10)) # create grid of frames
        vclip.append(batched_grid)
    vclip = torch.stack(vclip) # make grid of frames to grid of videos
    vclip = vclip.permute(0, 2, 3, 1) # T x C x H x W -> T x H x W x C
    return vclip    
    
def labels_grid(vid_shape,labels,font):
    tmps = []
    for t in range(vid_shape[0]):
        tmp = Image.new("RGB",(vid_shape[-2],vid_shape[-1]), (0,0,0))
        draw = ImageDraw.Draw(tmp)
        draw.text((0, 0),labels[t],(255,255,255),font=font)
        tmp = torchvision.transforms.functional.pil_to_tensor(tmp)
        tmps.append(tmp.int())                                                        
    return vutils.make_grid(tmps,normalize=False, scale_each=True, nrow=int(10)) 