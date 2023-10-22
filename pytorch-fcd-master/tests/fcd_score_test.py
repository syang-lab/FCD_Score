#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 19:52:24 2023

@author: shanyang
"""

import numpy as np
import torch
import fcd_score

import skimage
import pathlib
import clip



def test_CLIP_Features():
    IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm', 'tif', 'tiff', 'webp'}
    
    model, preprocess = clip.load("ViT-B/32")
    path = pathlib.Path(skimage.data_dir)
    files = sorted([file for ext in IMAGE_EXTENSIONS for file in path.glob('*.{}'.format(ext))])
    batch_size=4
    num_workers=0
    device=torch.device('cpu')
    all_features=fcd_score.CLIP_Features(files, model, preprocess, batch_size, device, num_workers)
    assert all_features.size()==(24,512)
    return all_features
 


def test_Statistics():
    IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm', 'tif', 'tiff', 'webp'}
    
    model, preprocess = clip.load("ViT-B/32")
    path = pathlib.Path(skimage.data_dir)
    files = sorted([file for ext in IMAGE_EXTENSIONS for file in path.glob('*.{}'.format(ext))])
    batch_size=4
    num_workers=0
    device=torch.device('cpu')
    
    mu,sigma=fcd_score.Statistics(files, model, preprocess, batch_size, device, num_workers)
    return mu,sigma
    

def test_Save_Stats():
    model, preprocess = clip.load("ViT-B/32")
    paths = [skimage.data_dir,"./mu_sigma"]
    batch_size=4
    num_workers=0
    device=torch.device('cpu')
    
    fcd_score.Save_Stats(paths, model, preprocess, batch_size,device, num_workers)
    
    
def test_Frechet_Distance():
    mu1=np.zeros((512,))
    sigma1= np.eye(512)
    mu2=np.ones((512,))
    sigma2= np.eye(512)
    
    assert mu1.shape == mu2.shape
    assert sigma1.shape == sigma2.shape
    
    FID_score=fcd_score.Frechet_Distance(mu1, sigma1, mu2, sigma2, eps=1e-6)
    assert FID_score==np.sum((mu1 - mu2)**2)
    return FID_score
    

def test_Frechet_CLIP_Distance():
    
    model, preprocess = clip.load("ViT-B/32")
    paths = [skimage.data_dir,skimage.data_dir]
    batch_size=4
    num_workers=0
    device=torch.device('cpu')
    
    FCD_score=fcd_score.Frechet_CLIP_Distance(paths, model, preprocess, batch_size, device, num_workers)
    assert FCD_score==0    
    return FCD_score



if __name__ == '__main__':    

    all_features=test_CLIP_Features()
    mu,sigma=test_Statistics()
    test_Save_Stats()
    FCD_score=test_Frechet_Distance()
    FCD_score=test_Frechet_CLIP_Distance()

















    
