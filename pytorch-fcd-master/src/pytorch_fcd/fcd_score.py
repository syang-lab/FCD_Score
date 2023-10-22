import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import pathlib

from PIL import Image

import numpy as np
from scipy import linalg

import torch
import clip


IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm', 'tif', 'tiff', 'webp'}

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, files, preprocess=None):
        self.files = files
        self.preprocess = preprocess

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert('RGB')
        if self.preprocess is not None:
            img = self.preprocess(img)
        return img


def CLIP_Features(files, model, preprocess, batch_size, device, num_workers):
    """calculate CLIP features

    Params:
        -- files       : list of image files paths
        -- model       : instance of CLIP model
        -- preprocess  : image transformaion
        -- batch_size  : batch size
        -- device      : device to run calculations
        -- num_workers : number of parallel dataloader workers

    Returns:
        -- clip features: (all number of images, feature_dims) 
    
    """
    
    if batch_size > len(files):
        print(('Warning: batch size is bigger than the data size.Setting batch size to data size'))
        batch_size = len(files)
        
    model.eval()

    dataset = ImageDataset(files, preprocess)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=True,
                                             num_workers=num_workers)
    
    all_features=list()
    for batch in dataloader:
        image_batch = batch.to(device)

        with torch.no_grad():
            features = model.encode_image(image_batch).float()
            all_features.append(features)
            
    all_features=torch.cat(all_features,axis=0)
    
    return all_features


def Statistics(files, model, preprocess, batch_size, device, num_workers):
    """calculation statistics
    Params:
        -- files       : list of image files paths
        -- model       : instance of CLIP model
        -- preprocess  : image transformaion
        -- batch_size  : batch size
        -- device      : device to run calculations
        -- num_workers : number of parallel dataloader workers

    Returns:
        -- mu   : mean of images
        -- sigma: covariance matrix of images
    
    """
    
    all_features = CLIP_Features(files, model, preprocess, batch_size, device, num_workers)
    mu = np.mean(all_features.numpy(), axis=0) 
    sigma = np.cov(all_features.numpy(), rowvar=False)
    return mu, sigma


def Compute_Statistics(path, model, preprocess, batch_size,device, num_workers):
    if path.endswith('.npz'):
        with np.load(path) as f:
            m, s = f['mu'][:], f['sigma'][:]
    else:
        path = pathlib.Path(path)
        files = sorted([file for ext in IMAGE_EXTENSIONS for file in path.glob('*.{}'.format(ext))])
        m, s = Statistics(files, model, preprocess, batch_size, device, num_workers)

    return m, s


def Save_Stats(savepath, model, preprocess, batch_size, device, num_workers):
    if not os.path.exists(savepath[0]):
        raise RuntimeError('Invalid path: %s' % savepath[0])

    if os.path.exists(savepath[1]):
        raise RuntimeError('Existing output file: %s' % savepath[1])


  
    m, s = Compute_Statistics(savepath[0], model, preprocess, batch_size, device, num_workers)
    np.savez_compressed(savepath[1], mu=m, sigma=s)
    return


def Frechet_Distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """ calculate Frechet Distance.
    Params:
        -- mu1   : mean of generaged images
        -- mu2   : mean of training dataset
        -- sigma1: covariance matrix of generated images
        -- sigma2: covariance matrix of training dataset

    Returns:
        -- Frechet Distance
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape
    assert sigma1.shape == sigma2.shape

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        print('FCD calculation produces singular product; adding %s to diagonal of cov estimates') % eps

        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))


    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    
    score=diff.dot(diff) + np.trace(sigma1)+ np.trace(sigma2) - 2 * tr_covmean
    return round(score,2)


def Frechet_CLIP_Distance(paths, model, preprocess, batch_size, device, num_workers):
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)


    m1, s1 = Compute_Statistics(paths[0], model, preprocess, batch_size, device, num_workers)
    m2, s2 = Compute_Statistics(paths[1], model, preprocess, batch_size, device, num_workers)
    
    FCD_score = Frechet_Distance(m1, s1, m2, s2)

    return FCD_score


def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model-name', type=str)
    parser.add_argument('--batch-size', type=int,default=2)
    parser.add_argument('--num-workers', type=int,default=0)
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--savepath', type=str, nargs=2,default=None,
                        help=('generate an npz archive from a directory of samples. the first path is used as input and the second as output.'))

    parser.add_argument('--paths', type=str, nargs=2,default=None,
                        help=('paths to the generated images and to .npz statistic files, to calculate FCD score'))

    
    args = parser.parse_args()

    if args.model_name not in clip.available_models():
        print("model do not exist, please select in the following list", clip.available_models())
        return 
    
    if args.device is None:
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    else:
        device = torch.device(args.device)

    if args.num_workers is None:
        try:
            num_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            num_cpus = os.cpu_count()

        num_workers = min(num_cpus, 8) if num_cpus is not None else 0
    else:
        num_workers = args.num_workers

    
    model, preprocess = clip.load(args.model_name)
    
    if args.savepath:
        Save_Stats(args.savepath, model, preprocess, args.batch_size, device, num_workers)
        return 
    
    if args.paths:
        FCD_score = Frechet_CLIP_Distance(args.paths, model, preprocess, args.batch_size, device, num_workers)
        print('FCD: ', FCD_score)


if __name__ == '__main__':
    main()
