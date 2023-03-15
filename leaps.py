import torch
import torch.nn as nn
import torch.optim as optim
import collections
import torch.cuda.amp as amp
import random
import torch
import torchvision
import torchvision.utils as vutils
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import pandas as pd
import sys, os, random, math, json

from utils.utils import lr_cosine_policy, clip, denormalize, create_folder

from torchvision.io import read_video
from einops import rearrange, reduce

from tqdm import tqdm

class LEAPSFeatHook():
    def __init__(self, module, t_dim=None):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.t_dim = t_dim
        self.feats = {}
        self.coh = {}

    def hook_fn(self, module, input, output):
        nch = input[0].shape[1]
        if len(input[0].shape) > 3:
            mean = input[0].mean([2, 3, 4])
            t1 = input[0][:,:,1:,...].unsqueeze(2)
            t2 = input[0][:,:,:-1,...].unsqueeze(3)
            t = t1-t2
            t = rearrange(t,' B C T1 T2 ... -> B C ... T1 T2')
            t = torch.tril(t,-1) 
            t = 1.0 - t
            t[t<0.] = 0.     
            coh = torch.norm(input[0][:,:,1:,...]-input[0][:,:,:-1,...],1) + torch.norm(t,1)
            self.coh[coh.device] =  coh # multi-gpu
        else:
            mean = input[0].mean([1])
        feature = mean
        self.feature = feature
        self.feats[feature.device] = feature # multi-gpu
        
    def close(self):
        self.hook.remove()


class VerFeatHook():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.feat_glob = {}

    def hook_fn(self, module, input, output):
        nch = input[0].shape[1]
        if len(input[0].shape) > 4: # 3D CNN
            nch = input[0].shape[1]
            mean = input[0].mean([0, 2, 3,4])
            var = input[0].permute(1, 0, 2, 3,4).contiguous().view([nch, -1]).var(1, unbiased=False)
        else: # 2D CNN
            nch = input[0].shape[1]
            mean = input[0].mean([0, 2, 3])
            var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)        
        feat = torch.norm(module.running_var.data - var, 2) + torch.norm(
            module.running_mean.data - mean, 2)
        self.feat = feat        
        self.feat_glob[feat.device] = feat # multi-gpu
        
    def close(self):
        self.hook.remove()


class LEAPS(object):
    def __init__(self, 
                 net=None,
                 net_verifier=None,
                 path="./generations/",
                 parameters=dict(),
                 coefficients=dict(),
                 gpus=[i for i in range(torch.cuda.device_count())],
                 iterations=3000):        
        torch.manual_seed(torch.cuda.current_device())
        self.gpus = gpus
        self.labels_csv = 'labels/validate.csv'
        self.class_ids = 'labels/kinetics_class_ids.json'
        self.store = True
        self.iterations = iterations
        self.num_frames = parameters["num_frames"]
        self.frame_resolution = parameters["resolution"]
        self.random_label = parameters["random_label"]
        self.do_flip = parameters["do_flip"]
        self.store_best = parameters["store_best"]
        self.kinetics_dir = parameters["stimuli_dir"]
        self.batch_size = parameters["batch_size"]
        self.use_fp16 = parameters["fp16"]
        self.criterion =parameters["critirion"]
        self.hook_for_display = parameters["hook_for_display"]
        self.save_every = 100
        self.jitter = 30
        self.bn_reg_scale = coefficients["reg"]
        self.first_bn_multiplier = 10.
        self.l2_scale = 0.005
        self.lr = coefficients["lr"]
        self.main_loss_multiplier = 1.0
        self.priming_coef = coefficients["prompt"] 
        self.num_generations = 0
        prefix = path
        self.prefix = prefix
        
        create_folder(prefix)
        create_folder(prefix + "/clips/")
        create_folder(prefix + "/best_clips/")
        create_folder(prefix + "/prompts/")
        create_folder(prefix + "/prompts/individual")        

        self.verfeats = []
        self.invertedfeats = []
        print(' \n Contstructing hooks ...')
        for module in net.modules():
            if isinstance(module, nn.BatchNorm3d) or isinstance(module, nn.LayerNorm):
                self.invertedfeats.append(LEAPSFeatHook(module))
                
        for module in net_verifier.modules():
            if isinstance(module, nn.BatchNorm3d) or isinstance(module, nn.LayerNorm):
                self.verfeats.append(VerFeatHook(module))
        
        self.net = torch.nn.DataParallel(net, device_ids=self.gpus)
        self.net_verifier = torch.nn.DataParallel(net, device_ids=self.gpus)
        print('Created a total of {} hooks'.format(len(self.invertedfeats)))
        print('Created a total of {} verifier hooks \n'.format(len(self.verfeats)))        
            

    
    def get_frames(self, targets=None):
        print("... Call to frame generation ...")
        net = self.net
        use_fp16 = self.use_fp16
        save_every = self.save_every
        saved_prompts = False
        current_device = torch.cuda.current_device()
        best_cost = 1e4
        criterion = self.criterion        
        kinetics_l = {}
        with open(self.class_ids) as json_file:
            kinetics_labels = json.load(json_file)
        # flip names with ids
        for k,v in kinetics_labels.items():
            kinetics_l[v] = k
        df_val = pd.read_csv(self.labels_csv)
        if targets is None:
            targets = torch.LongTensor([random.randint(0, 399) for _ in range(self.batch_size)]).to('cuda')
            if not self.random_label:
                targets=[169]   
                t_indx = 0
                new_targets = []
                while t_indx < self.batch_size:
                    list_idx = t_indx
                    while list_idx >= len(targets):
                        list_idx -= len(targets)
                    t_indx += 1  
                    new_targets.append(targets[list_idx])
                targets = new_targets
                targets = torch.LongTensor(targets * (int(self.batch_size / len(targets)))).to('cuda')     
                target_names = [kinetics_l[t.item()] for t in targets]
        matches = {}
        for t_s,t_i in zip(target_names,targets.cpu().numpy()):
            df = df_val[df_val['label'] == t_s]
            files = [os.path.join(self.kinetics_dir,'{}_{:06d}_{:06d}.mp4'.format(v['youtube_id'],int(v['time_start']),int(v['time_end']))) for _,v in df.iterrows()]
            # check that the file exists
            existing_files = []
            for file in files:
                if os.path.isfile(file):
                    existing_files.append(file)
            matches[t_i] = existing_files
        selected_videos = []
        for t_i in targets:
            is_above = False
            while not is_above:
                vi = random.choice(matches[t_i.cpu().item()])
                rvid = read_video(str(vi), output_format="TCHW")[0]
                rvid = torchvision.transforms.CenterCrop(self.frame_resolution)(rvid)
                frame_ids = list(range(0,rvid.shape[0]//1,math.ceil((rvid.shape[0]//1)/self.num_frames)))
                rvid = rvid[frame_ids][:self.num_frames] 
                rvid = rvid.unsqueeze(0).float()
                rvid = rvid.permute(0,2,1,3,4)
                rvid = rvid/255.
                rvid.requires_grad=False
                with torch.no_grad():
                    probs = net(rvid)
                #if not vi in selected_videos:
                is_above = True
                print('Found prompt {} for class {} with accuracy {}'.format(vi, t_i,probs[0][t_i].item()))
            selected_videos.append(vi)
        # create batch of masked videos
        batched_video_frames = [read_video(str(video_path), output_format="TCHW")[0] for video_path in selected_videos]
        vids = [torchvision.transforms.CenterCrop(self.frame_resolution)(v) for v in batched_video_frames] # spatial cropping
        frame_ids = [ list(range(0,v.shape[0]//1,math.ceil((v.shape[0]//1)/self.num_frames))) for v in vids]
        new_vids = [v[frame_ids[i]][:self.num_frames] for i,v in enumerate(vids)] 
        vids = torch.stack(new_vids).float()
        vids = vids.permute(0,2,1,3,4)
        vids = vids/255.
        vids.requires_grad=False
        
        img_original = self.frame_resolution
        num_frames = self.num_frames
        data_type = torch.float
        inputs = torch.randn((self.batch_size, 3, num_frames, img_original, img_original), requires_grad=True, device='cuda', dtype=data_type)
        
        #pooling_function = nn.Upsample(size=(self.num_frames,self.frame_resolution,self.frame_resolution),mode='trilinear')
        pooling_function = nn.Identity()
    
        for lr_it, lower_res in enumerate([1]):
    
            lim_0, lim_1 = self.jitter // lower_res, self.jitter // lower_res    
            optimizer = optim.Adam([inputs], lr=self.lr, betas=[0.5, 0.9], eps = 1e-8)
            do_clip = True
            # Creates a GradScaler once at the beginning of training.
            scaler = torch.cuda.amp.GradScaler()
            lr_scheduler = lr_cosine_policy(self.lr, 100, self.iterations)
            
            if use_fp16:
                dtype = torch.float16
            else:
                dtype = torch.float32
            
            with torch.autocast(device_type='cuda', dtype=dtype): 
                with torch.no_grad():
                    true_outs = net(vids)
                    prompt_saved_feats = []
                    for idx,layer in enumerate(self.invertedfeats):
                        vals = []
                        for value in layer.feats.values():
                            vals.append(value.clone().to('cuda'))
                        vals = torch.concat(vals,dim=0)
                        prompt_saved_feats.append(vals)
              
            for iteration in tqdm(range(self.iterations)):
                 
                # learning rate scheduling
                lr_scheduler(optimizer, iteration, iteration)
                inputs_jit = pooling_function(inputs)
                off1 = random.randint(-lim_0, lim_0)
                off2 = random.randint(-lim_1, lim_1)
                inputs_jit = torch.roll(inputs_jit, shifts=(off1, off2), dims=(-2, -1))
                
                # Flipping
                flip = random.random() > 0.5
                if flip and self.do_flip:
                    inputs_jit = torch.flip(inputs_jit, dims=(-1,))

                # forward pass
                optimizer.zero_grad()
                net.zero_grad()
                self.net_verifier.zero_grad()
                
                with torch.autocast(device_type='cuda', dtype=dtype): 
                    
                    outputs = net(inputs_jit)
                    v_out = self.net_verifier(inputs_jit)        
                    
                    # Cross entropy
                    loss = criterion(outputs, targets)
         
                    # Feature diversity regularisation
                    rescale = [self.first_bn_multiplier] +[1. + (self.first_bn_multiplier/(_s+1))  for _s in range(len(self.verfeats)-1)]
                        
                    diversity_reg = 0
                    for idx,layer in enumerate(self.verfeats):
                        l_g = layer.feat_glob
                        diversity_layer = []
                        for value in l_g.values():
                            diversity_layer.append(value.to('cuda') * rescale[idx])
                        diversity_layer = sum(diversity_layer)
                        diversity_reg = diversity_reg + diversity_layer
                    
                    rescale = [self.first_bn_multiplier] +[1. + (self.first_bn_multiplier/(_s+1))  for _s in range(len(self.invertedfeats)-1)]
                    
                    # Prompt feature loss
                    for idx,layer in enumerate(self.invertedfeats):
                        l_g = layer.feats
                        prompt_feats_layer_loss = []
                        vals = []
                        for value in layer.feats.values():
                            vals.append(value.clone().to('cuda'))
                        vals = torch.concat(vals,dim=0)
                        prompt_feats_layer_loss.append(abs(vals-prompt_saved_feats[idx]) * rescale[idx])
                    
                    cohs = 0.
                    for c in layer.coh.values():
                        cohs+=c.clone().to('cuda')
                    cohs /= self.batch_size
                
                    loss_prompt = sum(prompt_feats_layer_loss)
                    loss_prompt = torch.sum(loss_prompt)
                    
                    # l2 loss on images
                    loss_l2 = torch.norm(inputs_jit.view(self.batch_size, -1), dim=1).mean()
                            
                    # combining losses
                    loss_aux = self.bn_reg_scale * diversity_reg + \
                        self.l2_scale * loss_l2 + \
                        self.priming_coef * loss_prompt- \
                        1e-4 * cohs 
                    loss = self.main_loss_multiplier * loss + loss_aux
                    
                
                if iteration % save_every==0:
                    print("------------iteration {}----------".format(iteration))
                    print("total loss", loss.item())
                    print("diversity_reg", diversity_reg.item())
                    print("main criterion", criterion(outputs, targets).item())

                    if self.hook_for_display is not None:
                        acc = self.hook_for_display(inputs, targets)

                # do update
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
               
                # clip colors
                if do_clip:
                    inputs.data = clip(inputs.data, use_fp16=use_fp16)
                if best_cost > loss.item() or iteration == 1:
                    best_inputs = inputs.clone()
                    best_cost = loss.item()
                    
                if self.store:
                    if iteration % save_every==0: #and (save_every > 0):
                        
                        vclip = []
                        with open('{}/clips/ids.txt'.format(self.prefix), 'w+') as f:
                            f.write(str(targets.cpu().numpy()))
                        for t in range(inputs.shape[2]):
                            batched_grid = vutils.make_grid(denormalize(inputs[:,:,t,:,:].cpu()),normalize=False, scale_each=True, nrow=int(10))
                            vclip.append(batched_grid)
                        vclip = torch.stack(vclip)
                        vclip = vclip.permute(0, 2, 3, 1)     
                        
                        vvid = []
                        for t in range(vids.shape[2]):
                            batched_grid_vids= vutils.make_grid(vids[:,:,t,:,:].cpu(),normalize=True, scale_each=True, nrow=int(10))
                            vvid.append(batched_grid_vids)
                        vvid = torch.stack(vvid)
                        vvid = vvid.permute(0, 2, 3, 1)                  
                        
                        # create empty B images in PIL
                        max_size = 0
                        max_text = ''
                        for t in target_names:
                            if len(t) > max_size:
                                max_size = len(t)
                                max_text = t
                        fontsize = 1
                        font = ImageFont.truetype("DejaVuSans.ttf", fontsize)
                        while font.getsize(max_text)[0] < 0.85*inputs.shape[-1]:
                            # iterate until the text size is just larger than the criteria
                            fontsize += 1
                            font = ImageFont.truetype("DejaVuSans.ttf", fontsize)

                        tmps = []
                        for t in range(inputs.shape[0]):
                            tmp = Image.new("RGB",(inputs.shape[-2],inputs.shape[-1]), (0,0,0))
                            draw = ImageDraw.Draw(tmp)
                            draw.text((0, 0),target_names[t],(255,255,255),font=font)
                            tmp = torchvision.transforms.functional.pil_to_tensor(tmp)
                            tmps.append(tmp.int())                            
                            
                        tmps_grid = vutils.make_grid(tmps,normalize=False, scale_each=True, nrow=int(10))        
                        
                        vclip*=255
                        vclip += tmps_grid.permute(1,2,0).unsqueeze(0)
                        vclip[vclip>255] = 255.
                        
                        v_file =  '{}/clips/output_{:05d}_gpu_{}.mp4'.format(self.prefix,
                                                                            iteration // save_every,
                                                                            current_device)
                        vv_file =  '{}/prompts/output_{:05d}.mp4'.format(self.prefix,iteration // save_every)
                        v_file_f =  '{}/clips/output_{:05d}.mp4'.format(self.prefix,
                                                                        iteration // save_every)
                        torchvision.io.write_video(v_file, vclip,fps=5)
                        torchvision.io.write_video(vv_file, vvid*255,fps=5)
                        command =  "ffmpeg -y -hide_banner -loglevel error -stream_loop 3 -i {0} -c copy {1}".format(v_file, v_file_f)
                        os. system(command)
                        os.system('rm {}'.format(v_file))
                        
                        
                        for vidx,v in enumerate(vids):
                            if saved_prompts:
                                break
                            to_save_vid = v.cpu().permute(1, 2, 3, 0)   
                            to_save_vid *= 255.
                            vi_file = "{}/prompts/individual/out_{}.mp4".format(self.prefix,vidx)
                            vi_file_looped = "{}/prompts/individual/out_{}_looped.mp4".format(self.prefix,vidx)
                            torchvision.io.write_video(vi_file, to_save_vid,fps=5)
                            command =  "ffmpeg -y -hide_banner -loglevel error -stream_loop 3 -i {0} -c copy {1}".format(vi_file, vi_file_looped)
                            os. system(command)
                            os.system('rm {}'.format(vi_file))
                        saved_prompts = True
                        
            if self.store:
                if self.store_best:
                    vclip = []
                    with open('{}/best_clips/ids.txt'.format(self.prefix), 'w+') as f:
                        f.write(str(targets.cpu().numpy()))
                    #best_inputs = best_inputs.permute(0, 2, 3, 4, 1)     
                    vclip = denormalize(best_inputs).cpu()           
                    
                    # create empty B images in PIL
                    max_size = 0
                    max_text = ''
                    for t in target_names:
                        if len(t) > max_size:
                            max_size = len(t)
                            max_text = t
                    fontsize = max_size
                    font = ImageFont.truetype("DejaVuSans.ttf", fontsize)
                    while font.getsize(max_text)[0] < 0.75*vclip.shape[-1]:
                        # iterate until the text size is just larger than the criteria
                        fontsize += 1
                        font = ImageFont.truetype("DejaVuSans.ttf", fontsize)

                    tmps = []
                    for t in range(vclip.shape[0]):
                        tmp = Image.new("RGB",(inputs.shape[-2],vclip.shape[-1]), (0,0,0))
                        draw = ImageDraw.Draw(tmp)
                        #draw.text((0, 0),target_names[t],(255,255,255),font=font)
                        tmp = torchvision.transforms.functional.pil_to_tensor(tmp)
                        tmps.append(tmp.int())                            
                            
                    vclip*=255
                    best_vids = []
                    for idx,c in enumerate(vclip):
                        t = c+tmps[idx].unsqueeze(1)
                        t[t>255] = 255.
                        best_vids.append(t.detach().cpu())
                    
                    for i,v in enumerate(best_vids):
                        name = target_names[i].replace(' ','_')
                        name = name.replace('(','')
                        name = name.replace(')','')
                        
                        v_file =  '{}/best_clips/output_{}_{:05d}_1.mp4'.format(self.prefix,name,i)
                        v_file_f =  '{}/best_clips/output_{}_{:05d}.mp4'.format(self.prefix,name,i)
                        torchvision.io.write_video(v_file, v.permute(1,2,3,0),fps=5)
                        command =  "ffmpeg -y -hide_banner -loglevel error -stream_loop 3 -i {0} -c copy {1}".format(v_file, v_file_f)
                        os.system(command)
                        os.system('rm {}'.format(v_file))

        # to reduce memory consumption by states of the optimizer we deallocate memory
        optimizer.state = collections.defaultdict(dict)
            
        return loss.item(), criterion(outputs, targets).item() ,acc

    

    def synth(self, targets=None):
        if targets is not None:
            targets = torch.from_numpy(np.array(targets).squeeze()).cuda()
        losses = self.get_frames(targets=targets)
        self.num_generations += 1

        return losses