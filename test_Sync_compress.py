import os
import random
import time
import warnings
from collections import OrderedDict
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models

from utils.metric_func import *
from utils.util_func import *
from data.middlebury_dataloader_compress import Middlebury_dataset
from data.nyu_dataloader_compress import CompreNYU_v2_datset
from data.rgbdd_dataloader_compress import RGBDD_Dataset
from data.tofdsr_dataloader_compress import TOFDSR_Dataset
from PIL import Image
from tqdm import tqdm
from config import args as args_config
from model_list import import_model

args = args_config
best_rmse = 10.0


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus[0])
    val_Lu = Middlebury_dataset(root_dir='/opt/data/private/DSR/datasets/Lu/', scale=args.scale)
    val_Midd = Middlebury_dataset(root_dir='/opt/data/private/DSR/datasets/Middlebury/', scale=args.scale)
    val_NYU = CompreNYU_v2_datset(root_dir="/opt/data/private/DSR/datasets/nyu_data/", scale=args.scale,
                            train=False)
    test_minmax = np.load('%s/test_minmax.npy' % "/opt/data/private/DSR/datasets/nyu_data/")
    val_RGBDD = RGBDD_Dataset(root_dir="/opt/data/private/DSR/datasets/RGBDD/", scale=args.scale, downsample='sync',
                              train=False)
    val_TOFDSR = TOFDSR_Dataset(root_dir="/opt/data/private/DSR/datasets/", scale=args.scale, downsample='sync',
                                  train=False, txt_file="./data/TOFDC_Filled_Test.txt")

    model = import_model(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.pretrain is not None:
        print("Pretrain Paramter Path:", args.pretrain)
        checkpoint = torch.load(args.pretrain)
        try:
            loaded_state_dict = checkpoint['state_dict']
        except:
            loaded_state_dict = checkpoint
        new_state_dict = OrderedDict()
        for n, v in loaded_state_dict.items():
            name = n.replace("module.", "")
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        model = model.cuda()
        print('Load pretrained weight')

    print('MaxDepth: {} | H,W: {},{}'.format(args.max_depth, args.patch_height, args.patch_width))

    val_Lu_loaders = torch.utils.data.DataLoader(val_Lu, batch_size=1, shuffle=False, num_workers=4, pin_memory=False,
                                                 drop_last=False)
    val_Midd_loaders = torch.utils.data.DataLoader(val_Midd, batch_size=1, shuffle=False, num_workers=4,
                                                   pin_memory=False, drop_last=False)
    val_NYU_loaders = torch.utils.data.DataLoader(val_NYU, batch_size=1, shuffle=False, num_workers=4, pin_memory=False,
                                                  drop_last=False)
    val_RGBDD_loaders = torch.utils.data.DataLoader(val_RGBDD, batch_size=1, shuffle=False, num_workers=4,
                                                    pin_memory=False,
                                                    drop_last=False)
    val_TOFDSR_loaders = torch.utils.data.DataLoader(val_TOFDSR, batch_size=1, shuffle=False, num_workers=4,
                                                   pin_memory=False, drop_last=False)

    Lu_rmse, Lu_mse, Lu_mae, Lu_DEL_105, Lu_DEL_115, Lu_DEL_125 = test(val_Lu_loaders, model, dataname="Lu", minmax=None)
    Midd_rmse, Midd_mse, Midd_mae, Midd_DEL_105, Midd_DEL_115, Midd_DEL_125 = test(val_Midd_loaders, model, dataname="Midd", minmax=None)
    NYU_rmse, NYU_mse, NYU_mae, NYU_DEL_105, NYU_DEL_115, NYU_DEL_125 = test(val_NYU_loaders, model, dataname="NYU", minmax=test_minmax)
    RGBDD_rmse, RGBDD_mse, RGBDD_mae, RGBDD_DEL_105, RGBDD_DEL_115, RGBDD_DEL_125 = test(val_RGBDD_loaders, model, dataname="RGBDD", minmax=None)
    TOFDSR_rmse, TOFDSR_mse, TOFDSR_mae, TOFDSR_DEL_105, TOFDSR_DEL_115, TOFDSR_DEL_125 = test(val_TOFDSR_loaders, model, dataname="TOFDSR", minmax=None)

    print("===========================================Results============================================")
    print('NYU --> RMSE: %.2f, MSE: %.2f, MAE: %.2f, DEL_105: %.4f, DEL_115: %.4f, DEL_125: %.4f' % (
       NYU_rmse, NYU_mse, NYU_mae, NYU_DEL_105, NYU_DEL_115, NYU_DEL_125))
    print("----------------------------------------------------------------------------------------------")
    print('Lu --> RMSE: %.2f, MSE: %.2f, MAE: %.2f, DEL_105: %.4f, DEL_115: %.4f, DEL_125: %.4f' % (
       Lu_rmse, Lu_mse, Lu_mae, Lu_DEL_105, Lu_DEL_115, Lu_DEL_125))
    print("----------------------------------------------------------------------------------------------")
    print('Middlebury --> RMSE: %.2f, MSE: %.2f, MAE: %.2f, DEL_105: %.4f, DEL_115: %.4f, DEL_125: %.4f' % (
       Midd_rmse, Midd_mse, Midd_mae, Midd_DEL_105, Midd_DEL_115, Midd_DEL_125))
    print("----------------------------------------------------------------------------------------------")
    print('RGBDD --> RMSE: %.2f, MSE: %.2f, MAE: %.2f, DEL_105: %.4f, DEL_115: %.4f, DEL_125: %.4f' % (
        RGBDD_rmse, RGBDD_mse, RGBDD_mae, RGBDD_DEL_105, RGBDD_DEL_115, RGBDD_DEL_125))
    print("----------------------------------------------------------------------------------------------")
    print('TOFDSR --> RMSE: %.2f, MSE: %.2f, MAE: %.2f, DEL_105: %.4f, DEL_115: %.4f, DEL_125: %.4f' % (
       TOFDSR_rmse, TOFDSR_mse, TOFDSR_mae, TOFDSR_DEL_105, TOFDSR_DEL_115, TOFDSR_DEL_125))
    print("===============================================================================================")


def test(test_loader, model, dataname, minmax):
    rmse = AverageMeter('RMSE', ':.4f')
    mse = AverageMeter('MSE', ':.4f')
    mae = AverageMeter('MAE', ':.4f')
    DEL_105 = AverageMeter('DEL_105', ':.4f')
    DEL_115 = AverageMeter('DEL_115', ':.4f')
    DEL_125 = AverageMeter('DEL_125', ':.4f')
    model.eval()

    with torch.no_grad():
        for i, sample0 in enumerate(test_loader):
            sample = {key: val.to('cuda') for key, val in sample0.items() if (val is not None) and (key != 'name')}
            output = model(sample)
            rmse_result = 0.0
            mse_result = 0.0
            mae_result = 0.0
            DEL_105_result = 0.0
            DEL_115_result = 0.0
            DEL_125_result = 0.0

            out = output['pred']

            if dataname == "Lu":
                rmse_result = midd_calc_rmse(sample, output['pred'][0, 0])
                mse_result = midd_calc_mse(sample, output['pred'][0, 0])
                mae_result = midd_calc_mae(sample, output['pred'][0, 0])
                DEL_105_result, DEL_115_result, DEL_125_result = midd_calc_del(sample, output['pred'][0, 0])

            elif dataname == "Midd":
                rmse_result = midd_calc_rmse(sample, output['pred'][0, 0])
                mse_result = midd_calc_mse(sample, output['pred'][0, 0])
                mae_result = midd_calc_mae(sample, output['pred'][0, 0])
                DEL_105_result, DEL_115_result, DEL_125_result = midd_calc_del(sample, output['pred'][0, 0])

            elif dataname == "NYU":
                minmax0 = minmax[:, i]
                minmax0 = torch.from_numpy(minmax0).cuda()
                rmse_result = nyu_calc_rmse(sample, output['pred'][0, 0], minmax0)
                mse_result = nyu_calc_mse(sample, output['pred'][0, 0], minmax0)
                mae_result = nyu_calc_mae(sample, output['pred'][0, 0], minmax0)
                DEL_105_result, DEL_115_result, DEL_125_result = nyu_calc_del(sample, output['pred'][0, 0], minmax0)

            elif dataname == "RGBDD":
                rmse_result = rgbdd_calc_rmse(sample, output['pred'][0, 0])
                mse_result = rgbdd_calc_mse(sample, output['pred'][0, 0])
                mae_result = rgbdd_calc_mae(sample, output['pred'][0, 0])
                DEL_105_result, DEL_115_result, DEL_125_result = rgbdd_calc_del(sample, output['pred'][0, 0])

            elif dataname == "TOFDSR":
                rmse_result = tofdsr_calc_rmse(sample, output['pred'][0, 0])
                mse_result = tofdsr_calc_mse(sample, output['pred'][0, 0])
                mae_result = tofdsr_calc_mae(sample, output['pred'][0, 0])
                DEL_105_result, DEL_115_result, DEL_125_result = tofdsr_calc_del(sample, output['pred'][0, 0])

            rmse.update(rmse_result, sample['gt'].size(0))
            mse.update(mse_result, sample['gt'].size(0))
            mae.update(mae_result, sample['gt'].size(0))
            DEL_105.update(DEL_105_result, sample['gt'].size(0))
            DEL_115.update(DEL_115_result, sample['gt'].size(0))
            DEL_125.update(DEL_125_result, sample['gt'].size(0))

            print(rmse_result)

    return rmse.avg, mse.avg, mae.avg, DEL_105.avg, DEL_115.avg, DEL_125.avg


if __name__ == '__main__':
    main()
