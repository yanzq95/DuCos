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
from PIL import Image
from tqdm import tqdm
from config import args as args_config
from model_list import import_model

args = args_config
best_rmse = 10.0

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus[0])

    if args.isNosiy == 0:
        from data.tofdsr_dataloader import TOFDSR_Dataset
        val_TOFDSR = TOFDSR_Dataset(root_dir="/opt/data/private/DSR/datasets/", scale=args.scale, downsample='real',
                                     train=False, txt_file="./data/TOFDC_Filled_Test.txt")

    else:
        from data.tofdsr_dataloader_Noisy import TOFDSR_Dataset
        val_TOFDSR = TOFDSR_Dataset(root_dir="/opt/data/private/DSR/datasets/", scale=args.scale, downsample='real',
                                     train=False, txt_file="./data/TOFDC_Filled_Test.txt", isBlur=True, isNoisy=True)

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

    val_TOFDSR_loaders = torch.utils.data.DataLoader(val_TOFDSR, batch_size=1, shuffle=False, num_workers=4,
                                                     pin_memory=False, drop_last=False)

    TOFDSR_rmse, TOFDSR_mse, TOFDSR_mae, TOFDSR_DEL_105, TOFDSR_DEL_115, TOFDSR_DEL_125 = test(val_TOFDSR_loaders, model, dataname="TOFDSR", minmax=None)

    print("===========================================Results============================================")
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

            out = output['pred']

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
