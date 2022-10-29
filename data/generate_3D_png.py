import os
import argparse
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
from functools import partial


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

def crop_center(img, crop_size=(2048,2048)):
    cm_x, cm_y = mass_center(FFT_gaussian(img))
    
    def correct_overlap(imlen=None, croplen=None, center=None):
        assert croplen <= imlen
        assert center>0 and center<imlen
        if min(imlen-center-croplen/2,center-croplen/2) >= 0:
            return center
        if imlen-center-croplen/2 < center-croplen/2:
            return imlen-croplen/2
        else:
            return croplen/2
    
    cm_x = correct_overlap(img.shape[0], crop_size[0], cm_x)
    cm_y = correct_overlap(img.shape[1], crop_size[1], cm_y)
    
#     return img[int(cm_x-crop_size[0]/2):int(cm_x+crop_size[0]/2) , int(cm_y-crop_size[1]/2):int(cm_y+crop_size[1]/2)] #either return the cropped image
    return cm_x,cm_y  #or return the crop paramters (necessary for 3D)

def crop(img, centroid, crop_size=(2048,2048)):
    cm_x, cm_y = centroid
    return img[int(cm_x-crop_size[0]/2):int(cm_x+crop_size[0]/2) , int(cm_y-crop_size[1]/2):int(cm_y+crop_size[1]/2)]

def create_imstack(impaths, downsample = 1, do_crop = (2048,2048), do_norm = True, isotropic = True, savelocation = None):
    
    firstpath, ext = os.path.splitext(impaths[0])
    print('{}'.format(firstpath))
    if ext == '.DCM':
        imlist = [dicom.dcmread(impath).pixel_array for impath in impaths]
    elif ext == '.png' or ext == '.PNG' or ext == '.jpg' or ext == '.tif':
        imlist = [np.array(Image.open(impath)) for impath in impaths]
    else:
        raise AssertionError ('unknown file extension')
        
    savepaths = [impath.replace(ext, '.png') for impath in impaths]
    
    if downsample != 1:
        print('downsample not yet implemented')#-------------------------use torchvision transforms for a downsampler?--------------------------
    if do_norm:
        imlist = [normalize(img) for img in imlist]
    if do_crop is not None:
        crops = [crop_center(img, crop_size=do_crop) for img in imlist[::10]]
        x = sum([i[0] for i in crops])/len(crops)
        y = sum([i[1] for i in crops])/len(crops)
        imlist = [crop(img, crop_size=do_crop, centroid = (x,y)) for img in imlist]
    
    firstpath = os.path.basename(firstpath)
    if os.path.exists('{}/{}'.format(savelocation, firstpath)) == False:
        os.makedirs('{}/{}'.format(savelocation, firstpath))
    for i in range(len(imlist)):
#         imsave(savelocation+'/'+firstpath+'/'+os.path.basename(savepaths[i]),imlist[i])
        imsave('{}/{}/{}'.format(savelocation, firstpath, os.path.basename(savepaths[i])), imlist[i])
    

# def save_to_tiff(impath, do_norm=True, do_crop=(2048,2048),savepath = None):
#     img = dicom.dcmread(impath).pixel_array
#     if savepath is None:
#         savepath = impath.replace('.DCM', '.png')
#     if do_norm:
#         img = normalize(img)
#     if do_crop is not None:
#         img = crop(img)
#     imsave(savepath,img)
#     print(savepath)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='create 3D image stacks')
    parser.add_argument('--input_directory')
    parser.add_argument('--zheight', type=int, default=128)
    parser.add_argument('--normalize', type=bool, default=True)
    parser.add_argument('--crop', nargs="+", type=int,default=(2048,2048))
    parser.add_argument('--downsample', nargs="+", type=int, default=1)
    parser.add_argument('--isotropic', type=float, default=1)
    parser.add_argument('--output_directory', default=None)
    args = parser.parse_args()
    
    args.crop = tuple(args.crop)
        
    if args.output_directory is None:
        args.output_directory=args.input_directory
    print('\ngenerating stacks of height {}'.format(args.zheight))
    
    impaths = sorted(glob.glob('{}/*.DCM'.format(args.input_directory)))
    num_stacks = len(impaths)//args.zheight
    print('number of complete zstacks = {}, remainder = {}\n'.format(num_stacks, len(impaths)%args.zheight))
    stackpaths = []
    for i in range(num_stacks):
        stackpaths.append(impaths[i*args.zheight:(i+1)*args.zheight])
    
    if args.output_directory is None:
        args.output_driectory = '{}_{}x{}x{}'.format(args.input_directory,args.zheight,args.crop[0],args.crop[1])
    else:
        args.output_directory = '{}_{}x{}x{}'.format(args.output_directory,args.zheight,args.crop[0],args.crop[1])
    if os.path.exists(args.output_directory) == False:
        os.makedirs(args.output_directory)
    
    partialfunc = partial(create_imstack,
                         downsample=args.downsample, 
                         do_crop=args.crop, 
                         do_norm=args.normalize, 
                         isotropic = args.isotropic,
                         savelocation = args.output_directory,
                        )
    
    with mp.Pool(mp.cpu_count()) as pool:
        pool.map(partialfunc, stackpaths)