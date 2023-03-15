import argparse
import torch
from torch import distributed, nn
import random
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torchvision import datasets, transforms
from typing import Union, Tuple
import numpy as np
import os, sys
import models.video_model_builder as model_builder
from utils.utils import load_model_pytorch, distributed_is_initialized


from leaps import LEAPS
from utils.utils import accuracy

random.seed(0)


def validate(input, target, model, use_fp16,eps=1E-16):
   
    with torch.no_grad():
        if use_fp16:
            dtype = torch.float16
        else:
            dtype = torch.float32
        if use_fp16:
            with torch.autocast(device_type='cuda', dtype=dtype):
                output = model(input)
                prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        else:
            output = model(input)
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        prob_y = nn.functional.softmax(output,dim=-1).data.cpu().numpy()# calculate prob y
    print("Verifier accuracy: {}".format(prec1.item()))


def run(args):
    assert args.stimuli_dir is not None, "Missing directory with stimuli videos"
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = model_builder.get_models_dict()
    assert args.model in models, 'Architecture not supported, please use one of the following models {}'.format(models.keys()) 
    cfg_net , net = models[args.model](num_frames=args.num_frames,frame_size=args.resolution)
    net = net.to(device)
    net.to(device)
    net.eval()
    if args.verifier is not None:
        verifier=args.verifier
    assert verifier in models, 'Verifier not found, available models include {}'.format(models.keys())
    print("loading verifier: ", verifier)
    cfg_verifier, net_verifier = models[verifier](num_frames=args.num_frames,frame_size=args.resolution)
    net_verifier = net_verifier.to(device)
    net_verifier.eval()
    
    store_dir = args.store_dir
    store_dir = "generations/{}".format(store_dir)

    parameters = dict()
    parameters["num_frames"] = cfg_net.DATA.NUM_FRAMES
    parameters["resolution"] = cfg_net.TRAIN_CROP_SIZE
    parameters["do_flip"] = args.do_flip
    parameters["random_label"] = args.random_label
    parameters["store_best"] = args.store_best
    parameters["stimuli_dir"] = args.stimuli_dir
    parameters["batch_size"] = args.batch_size
    parameters["fp16"] = args.fp16
    parameters["critirion"] = nn.CrossEntropyLoss()
    parameters["hook_for_display"] = lambda x,y: validate(x, y, net_verifier, args.fp16)
    coefficients = dict()
    coefficients["reg"] = args.reg
    coefficients["lr"] = args.lr
    coefficients["prompt"] = 1e-2

    learned_preconscious = LEAPS(net=net,
                                 net_verifier=net,
                                 path=store_dir,
                                 parameters=parameters,
                                 coefficients=coefficients,
                                 gpus = args.gpu_ids,
                                 iterations= args.iterations)
    
    
    learned_preconscious.synth()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', default=1500, type=int, help='number of iterations')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--stimuli_dir', default=None, type=str, help='directory of stimuli videos')
    parser.add_argument('--model', default='x3d-xs', type=str, help='model name')
    parser.add_argument('--fp16', default=True, type=bool, help='use mixed precision')
    parser.add_argument('--store_dir', type=str, default='test', help='folder to store synthesized videos: `generated/{folder name}`')
    parser.add_argument('--verifier', type=str, default='x3d_xs', help = "verifier model")
    parser.add_argument('--do_flip', type=bool, default=True, help='apply flip')
    parser.add_argument('--random_label', default=False, help='use random labels')
    parser.add_argument('--reg', type=float, default=0.05, help='coefficient for regularization')
    parser.add_argument('--lr', type=float, default=.6, help='learning rate')
    parser.add_argument('--store_best', default=True, help='save best results separately')
    parser.add_argument('--gpu_ids', nargs='+', type=int, help='ids of gpus')
    parser.add_argument('--resolution', type=int, default=None, help='frames resolution')
    parser.add_argument('--num_frames', type=int, default=None, help='number of frames')
    

    args = parser.parse_args()
    print(args)

    torch.backends.cudnn.benchmark = True
    run(args)


if __name__ == '__main__':
    main()
    
    
