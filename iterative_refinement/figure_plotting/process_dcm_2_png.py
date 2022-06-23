import pydicom as dicom
import cv2   
import glob
from tqdm import tqdm
from torchvision import transforms, utils
from PIL import Image, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
from itertools import islice 
from tifffile import imsave
from scipy import ndimage
import multiprocessing as mp
import os

def normalize(img, pxrange=(0,8000)):
    img = np.clip(img, pxrange[0], pxrange[1])
    img = np.divide(img, pxrange[1]-pxrange[0])
    img = img*255
    return img.astype(np.uint8) #return image in uint8 format for max memory

def FFT_gaussian(image, kernel = 500):
    #apply large kernel blur in frequency space
    input_ = np.fft.fft2(image)
    result = ndimage.fourier_gaussian(input_, sigma=kernel)
    result = np.fft.ifft2(result)
    return result.real

def mass_center(image):
    return np.unravel_index(np.argmax(image, axis=None), image.shape)

def crop(img, crop_size=(2048,2048), crop_center=None):
    if crop_center is not None:
        cm_y, cm_x = crop_center
#         print('cropped to {}, {}'.format(cm_x, cm_y))
        return img[int(cm_x-crop_size[0]/2):int(cm_x+crop_size[0]/2) , int(cm_y-crop_size[1]/2):int(cm_y+crop_size[1]/2)]
    
    cm_x, cm_y = mass_center(FFT_gaussian(img))
    
    def correct_overlap(imlen=None, croplen=None, center=None):
        assert croplen<imlen
        assert center>0 and center<imlen
        if min(imlen-center-croplen/2,center-croplen/2) >= 0:
            return center
        if imlen-center-croplen/2 < center-croplen/2:
            return imlen-croplen/2
        else:
            return croplen/2
    
    cm_x = correct_overlap(img.shape[0], crop_size[0], cm_x)
    cm_y = correct_overlap(img.shape[1], crop_size[1], cm_y)
    
    return img[int(cm_x-crop_size[0]/2):int(cm_x+crop_size[0]/2) , int(cm_y-crop_size[1]/2):int(cm_y+crop_size[1]/2)] #either return the cropped image
#     return int(cm_x-crop_size[0]/2), int(cm_x+crop_size[0]/2), int(cm_y-crop_size[1]/2), int(cm_y+crop_size[1]/2)  #or return the crop paramters (necessary for 3D)

def save_to_tiff(impath, crop_center, savepath, do_norm=True, do_crop=(2048,2048), resize=False):
    if savepath is None:
        savepath = impath.replace('.DCM', '.png')
#     if os.path.exists(savepath):
#         return 0
    img = dicom.dcmread(impath).pixel_array
    if do_norm:
        img = normalize(img)
    if do_crop is not None:
        img = crop(img, crop_center = crop_center)
    if resize is True:
        img = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    imsave(savepath,img)
#     print(savepath)
    
if __name__ == '__main__':
    dim = 2
    #FOR 2D only
    images_path = '../datasets/2022_02_FullDataset/2020/ND17348_689/*.DCM'
    with mp.Pool(mp.cpu_count()) as pool:
        pool.map(save_to_tiff, glob.glob(images_path))
