import torch

def rgb2xyz(var, device = 'cuda'):
    #input (min, max) = (0, 1)
    #output (min, max) = (0, 1)
    transform = torch.FloatTensor([[0.412453, 0.357580, 0.180423], 
                            [0.212671, 0.715160, 0.072169], 
                            [ 0.019334,  0.119193,  0.950227]]).to(device)
    xyz = torch.matmul(var, transform.t())
    return xyz

def rgb2ycrcb(imgs):
    #input (min, max) = (0, 1)
    #output (min, max) = (0, 1)
    r = imgs[..., 0] * 255
    g = imgs[..., 1] * 255
    b = imgs[..., 2] * 255
    y = 0.299*r + 0.587*g + 0.114*b
    cr = (r - y)*0.713 + 128
    cb = (b - y)*0.564 + 128
    ycrcb = torch.stack([y, cb, cr], -1)
    return (ycrcb - 16) / (240 - 16)

def rgb2srgb(imgs):
    return torch.where(imgs <= 0.04045, imgs/12.92, torch.pow((imgs + 0.055)/1.055, 2.4))

def rgb2cmyk(imgs, device='cuda'):
    r = imgs[..., 0]
    g = imgs[..., 1]
    b = imgs[..., 2]
    k = 1 - torch.max(imgs, dim=-1).values
    c = (1-r-k)/(1-k + 1e-7)
    m = (1-g-k)/(1-k + 1e-7)
    y = (1-b-k)/(1-k + 1e-7)
    result = torch.stack([c, m, y, k], -1).clamp(0, 1)
    return result

def convert_rgb(imgs, target='rgb'):
    if target == 'rgb':
        return imgs
    elif target == 'cmyk':
        return rgb2cmyk(imgs)
    elif target == 'xyz':
        return rgb2xyz(imgs)
    elif target == 'ycrcb':
        return rgb2ycrcb(imgs)
    elif target == 'srgb':
        return rgb2srgb(imgs)