import math
import numpy as np
import torch
import torch.nn.functional as F
import warnings


# ====================================TOFDSR Metric=========================================
def tofdsr_calc_rmse(sample, out):
    gt = sample['gt'][0, 0]
    maxx = sample['max']
    minn = sample['min']

    gt = gt[6:-6, 6:-6]
    out = out[6:-6, 6:-6]

    mask = (gt >= 100) & (gt <= 5000)
    gt = gt[mask]
    out = out[mask]

    out = out * (maxx - minn) + minn
    gt = gt / 10.0
    out = out / 10.0

    return torch.sqrt(torch.mean(torch.pow(gt - out, 2)))


def tofdsr_calc_mse(sample, out):
    gt = sample['gt'][0, 0]
    maxx = sample['max']
    minn = sample['min']

    gt = gt[6:-6, 6:-6]
    out = out[6:-6, 6:-6]

    mask = (gt >= 100) & (gt <= 5000)
    gt = gt[mask]
    out = out[mask]

    out = out * (maxx - minn) + minn
    gt = gt / 10.0
    out = out / 10.0

    return torch.mean(torch.pow(gt - out, 2))


def tofdsr_calc_mae(sample, out):
    gt = sample['gt'][0, 0]
    maxx = sample['max']
    minn = sample['min']

    gt = gt[6:-6, 6:-6]
    out = out[6:-6, 6:-6]

    mask = (gt >= 100) & (gt <= 5000)
    gt = gt[mask]
    out = out[mask]

    out = out * (maxx - minn) + minn
    gt = gt / 10.0
    out = out / 10.0

    return torch.mean(torch.abs(gt - out))


def tofdsr_calc_del(sample, out):
    gt = sample['gt'][0, 0]
    maxx = sample['max']
    minn = sample['min']

    gt = gt[6:-6, 6:-6]
    out = out[6:-6, 6:-6]

    mask = (gt >= 100) & (gt <= 5000)
    gt = gt[mask]
    out = out[mask]
    out = out * (maxx - minn) + minn
    gt = gt / 10.0
    out = out / 10.0

    # For numerical stability
    num_valid = mask.sum()

    # delta
    r1 = gt / (out + 1e-8)
    r2 = out / (gt + 1e-8)
    ratio = torch.max(r1, r2)

    del_105 = (ratio < 1.05).type_as(ratio)
    del_115 = (ratio < 1.15).type_as(ratio)
    del_125 = (ratio < 1.25).type_as(ratio)

    del_105 = del_105.sum() / (num_valid + 1e-8)
    del_115 = del_115.sum() / (num_valid + 1e-8)
    del_125 = del_125.sum() / (num_valid + 1e-8)

    return del_105, del_115, del_125


# ====================================TOFDSR Metric=========================================

def syncrgbdd_calc_rmse(sample, out):
    gt = sample['gt'][0, 0]
    gt = gt[6:-6, 6:-6]
    out = out[6:-6, 6:-6]

    maxx = sample['max']
    minn = sample['min']

    gt = gt * (maxx - minn) + minn
    out = out * (maxx - minn) + minn

    gt = gt * 100
    out = out * 100

    return torch.sqrt(torch.mean(torch.pow(gt - out, 2)))


# ====================================Middlebury Metric=========================================
def midd_calc_rmse(sample, out):
    gt = sample['gt'][0, 0]
    gt = gt[6:-6, 6:-6]
    out = out[6:-6, 6:-6]
    gt = gt * 255.0
    out = out * 255.0

    return torch.sqrt(torch.mean(torch.pow(gt - out, 2)))

def midd_calc_mse(sample, out):
    gt = sample['gt'][0, 0]
    gt = gt[6:-6, 6:-6]
    out = out[6:-6, 6:-6]
    gt = gt * 255.0
    out = out * 255.0

    return torch.mean(torch.pow(gt - out, 2))

def midd_calc_mae(sample, out):
    gt = sample['gt'][0, 0]
    gt = gt[6:-6, 6:-6]
    out = out[6:-6, 6:-6]
    gt = gt * 255.0
    out = out * 255.0

    return torch.mean(torch.abs(gt - out))

def midd_calc_del(sample, out):
    gt = sample['gt'][0, 0]
    gt = gt[6:-6, 6:-6]
    out = out[6:-6, 6:-6]

    # For numerical stability+
    mask = gt >= 0.0
    num_valid = mask.sum()
    out = out[mask]
    gt = gt[mask]

    gt = gt * 255.0
    out = out * 255.0

    # delta
    r1 = gt / (out + 1e-8)
    r2 = out / (gt + 1e-8)
    ratio = torch.max(r1, r2)

    del_105 = (ratio < 1.05).type_as(ratio)
    del_115 = (ratio < 1.15).type_as(ratio)
    del_125 = (ratio < 1.25).type_as(ratio)

    del_105 = del_105.sum() / (num_valid + 1e-8)
    del_115 = del_115.sum() / (num_valid + 1e-8)
    del_125 = del_125.sum() / (num_valid + 1e-8)

    return del_105, del_115, del_125
# ====================================Middlebury Metric=========================================

# ====================================NYU Metric=========================================
def nyu_calc_rmse(sample, out, minmax):
    gt = sample['gt'][0, 0]
    gt = gt[6:-6, 6:-6]
    out = out[6:-6, 6:-6]

    gt = gt * (minmax[0] - minmax[1]) + minmax[1]
    out = out * (minmax[0] - minmax[1]) + minmax[1]
    gt = gt * 100
    out = out * 100

    return torch.sqrt(torch.mean(torch.pow(gt - out, 2)))


def nyu_calc_mse(sample, out, minmax):
    gt = sample['gt'][0, 0]
    gt = gt[6:-6, 6:-6]
    out = out[6:-6, 6:-6]

    gt = gt * (minmax[0] - minmax[1]) + minmax[1]
    out = out * (minmax[0] - minmax[1]) + minmax[1]
    gt = gt * 100
    out = out * 100

    return torch.mean(torch.pow(gt - out, 2))

def nyu_calc_mae(sample, out, minmax):
    gt = sample['gt'][0, 0]
    gt = gt[6:-6, 6:-6]
    out = out[6:-6, 6:-6]

    gt = gt * (minmax[0] - minmax[1]) + minmax[1]
    out = out * (minmax[0] - minmax[1]) + minmax[1]
    gt = gt * 100
    out = out * 100

    return torch.mean(torch.abs(gt - out))

def nyu_calc_del(sample, out, minmax):
    gt = sample['gt'][0, 0]
    gt = gt[6:-6, 6:-6]
    out = out[6:-6, 6:-6]

    mask = gt >= 0.0
    num_valid = mask.sum()
    out = out[mask]
    gt = gt[mask]

    gt = gt * (minmax[0] - minmax[1]) + minmax[1]
    out = out * (minmax[0] - minmax[1]) + minmax[1]
    gt = gt * 100
    out = out * 100

    # delta
    r1 = gt / (out + 1e-8)
    r2 = out / (gt + 1e-8)
    ratio = torch.max(r1, r2)

    del_105 = (ratio < 1.05).type_as(ratio)
    del_115 = (ratio < 1.15).type_as(ratio)
    del_125 = (ratio < 1.25).type_as(ratio)

    del_105 = del_105.sum() / (num_valid + 1e-8)
    del_115 = del_115.sum() / (num_valid + 1e-8)
    del_125 = del_125.sum() / (num_valid + 1e-8)

    return del_105, del_115, del_125
# ====================================NYU Metric=========================================

# ====================================RGBDD Metric=========================================
def rgbdd_calc_rmse(sample, out):
    gt = sample['gt'][0, 0]
    gt = gt[6:-6, 6:-6]
    out = out[6:-6, 6:-6]

    maxx = sample['max']
    minn = sample['min']
    out = out * (maxx - minn) + minn

    gt = gt / 10.0
    out = out / 10.0

    return torch.sqrt(torch.mean(torch.pow(gt - out, 2)))

def rgbdd_calc_mse(sample, out):
    gt = sample['gt'][0, 0]
    gt = gt[6:-6, 6:-6]
    out = out[6:-6, 6:-6]

    maxx = sample['max']
    minn = sample['min']
    out = out * (maxx - minn) + minn

    gt = gt / 10.0
    out = out / 10.0

    return torch.mean(torch.pow(gt - out, 2))

def rgbdd_calc_mae(sample, out):
    gt = sample['gt'][0, 0]
    gt = gt[6:-6, 6:-6]
    out = out[6:-6, 6:-6]

    maxx = sample['max']
    minn = sample['min']
    out = out * (maxx - minn) + minn

    gt = gt / 10.0
    out = out / 10.0

    return torch.mean(torch.abs(gt - out))

def rgbdd_calc_del(sample, out):
    gt = sample['gt'][0, 0]
    gt = gt[6:-6, 6:-6]
    out = out[6:-6, 6:-6]

    mask = gt >= 0.0
    num_valid = mask.sum()
    out = out[mask]
    gt = gt[mask]

    maxx = sample['max']
    minn = sample['min']
    out = out * (maxx - minn) + minn

    gt = gt / 10.0
    out = out / 10.0

    # delta
    r1 = gt / (out + 1e-8)
    r2 = out / (gt + 1e-8)
    ratio = torch.max(r1, r2)

    del_105 = (ratio < 1.05).type_as(ratio)
    del_115 = (ratio < 1.15).type_as(ratio)
    del_125 = (ratio < 1.25).type_as(ratio)

    del_105 = del_105.sum() / (num_valid + 1e-8)
    del_115 = del_115.sum() / (num_valid + 1e-8)
    del_125 = del_125.sum() / (num_valid + 1e-8)

    return del_105, del_115, del_125
# ====================================RGBDD Metric=========================================

def rmse_eval(sample, output, t_valid=1e-3):
    with torch.no_grad():
        pred = output.detach()
        gt = sample['gt'].detach()

        mask = gt > t_valid
        num_valid = mask.sum()

        pred = pred[mask]
        gt = gt[mask]

        diff = pred - gt
        diff_sqr = torch.pow(diff, 2)

        rmse = diff_sqr.sum() / (num_valid + 1e-8)
        rmse = torch.sqrt(rmse)

    return rmse

def mae_eval(sample, output, t_valid=1e-3):
    with torch.no_grad():
        pred = output.detach()
        gt = sample['gt'].detach()

        mask = gt > t_valid
        num_valid = mask.sum()

        pred = pred[mask]
        gt = gt[mask]

        diff = pred - gt
        diff_abs = torch.abs(diff)

        rmse = diff_abs.sum() / (num_valid + 1e-8)

    return rmse


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=False):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)
