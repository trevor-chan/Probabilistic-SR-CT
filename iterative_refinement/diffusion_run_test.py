import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from tensorboardX import SummaryWriter
import os
import numpy as np
from PIL import Image
from tqdm import tqdm

class args_standin():
    def __init__(self, config, phase, gpu_ids):
        self.config = config
        self.phase = phase
        self.gpu_ids = gpu_ids
        self.debug = False

# args = args_standin('config/mri_tibia.json', 'val', None)
args = args_standin('config/256_3_testing_config.json', 'val', None)


opt = Logger.parse(args)

opt['name'] = 'RUNNING_INFERENCE'
opt['path']['resume_state'] = 'experiments/256_test_220225_202536/checkpoint/I990000_E187'
# opt['path']['resume_state'] = 'experiments/256_test_220225_202536/checkpoint/I1000000_E189'
# opt['datasets']['val']['dataroot'] = 'data/datasets/all_pngs/test_85_256'
opt['datasets']['val']['dataroot'] = 'figure_plotting/test_85_256'


val_set = Data.create_dataset(opt['datasets']['val'], 'val')
val_loader = Data.create_dataloader(
        val_set, opt['datasets']['val'], 'val', batch_size=1)


diffusion = Model.create_model(opt)

diffusion.set_new_noise_schedule(opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])

def to_shape_2d(a, shape):
    y_, x_ = shape
    y, x = a.shape
    y_pad = (y_-y)
    x_pad = (x_-x)
    return np.pad(a,((y_pad//2, y_pad//2 + y_pad%2), 
                     (x_pad//2, x_pad//2 + x_pad%2)),
                  mode = 'edge')

datalen = len(val_loader.dataset)

def test_image(val_data, model, result_path, scalefactor):
    diffusion=model
    diffusion.feed_data(val_data)
    diffusion.test(continous=False)
    visuals = diffusion.get_current_visuals()
    sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
    hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
    lr_img = Metrics.tensor2img(visuals['LR'])  # uint8
    fake_img = Metrics.tensor2img(visuals['INF'])  # uint8
    lr_img = np.repeat(np.repeat(lr_img, scalefactor, axis=0), scalefactor, axis=1)
    #for if lr_img is not the same dimensions
    if lr_img.shape != fake_img.shape:
            lr_img = to_shape_2d(lr_img, fake_img.shape)
    out_img = np.concatenate((hr_img,lr_img,fake_img,sr_img),axis=1)
    Metrics.save_img(out_img, '{}.png'.format(result_path))
    return Metrics.calculate_psnr(sr_img, hr_img)


scalefactor = int(opt['datasets']['train']['r_resolution']/opt['datasets']['train']['l_resolution'])

avg_psnr = 0
count = 0
start_index = 0
end_index = 3499

for i,  val_data in enumerate(val_loader):
    if i < start_index:
        continue
    if i > end_index:
        break
    current_psnr = test_image(val_data, diffusion, "misc/85_256_outputs/test_{}_r0".format(i), scalefactor)
    avg_psnr += current_psnr
    count += 1
    print('image {0} / {1}: psnr = {2:.2f}'.format(i, datalen, current_psnr))
    
avg_psnr = avg_psnr / count
print('FINAL COUNT: {}'.format(count))
print('FINAL PSNR: {0:.4f}'.format(avg_psnr))
