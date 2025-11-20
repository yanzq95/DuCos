import csv
import os
import random
import time
import json
from tqdm import tqdm
import warnings
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from utils.util_func import *
from utils.metric_func import *

from config import args as args_config
from model_list import import_model
import kornia

warnings.filterwarnings('ignore')

args = args_config
best_rmse = 10.0
fieldnames = ['epoch', 'loss', 'rmse', 'mae', 'absrel', 'delta1_02', 'delta1_05', 'delta1_10', 'delta_125_1',
              'delta_125_2', 'delta_125_3']


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    args.workers = 4 * len(convert_str_to_num(args.gpus, 'int'))

    current_time = time.strftime('%y%m%d_%H%M%S')
    args.save_dir = './workspace/logs/train/{}_{}_{}'.format(current_time, args.model_name, args.save)
    os.makedirs(args.save_dir, exist_ok=True)
    print('Everything related to this model will be stored here {}'.format(args.save_dir))

    with open(args.save_dir + '/train.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    with open(args.save_dir + '/val.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    args.num_gpu = torch.cuda.device_count()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    main_worker(args)


def main_worker(args):
    global best_rmse
    best_rmse = 100.0
    args.num_images_seen = 0

    if args.data_name == 'RGBDD':
        if args.isNosiy == 0:
            from data.rgbdd_dataloader import RGBDD_Dataset
            args.max_depth = 10.0
            train_dataset = RGBDD_Dataset(root_dir="/opt/data/private/DSR/datasets/RGBDD/", scale=4, downsample='real',
                                          train=True)
            val_datasets = RGBDD_Dataset(root_dir="/opt/data/private/DSR/datasets/RGBDD/", scale=4, downsample='real',
                                         train=False)
        else:
            from data.rgbdd_dataloader_Noisy import RGBDD_Dataset
            args.max_depth = 10.0
            train_dataset = RGBDD_Dataset(root_dir="/opt/data/private/DSR/datasets/RGBDD/", scale=4, downsample='real',
                                          train=True, isBlur=True, isNoisy=True)
            val_datasets = RGBDD_Dataset(root_dir="/opt/data/private/DSR/datasets/RGBDD/", scale=4, downsample='real',
                                         train=False, isBlur=True, isNoisy=True)
        print('Dataset is RGBDD')
    else:
        print("Please Choice Dataset !!")
        raise NotImplementedError

    model = import_model(args)
    init_lr = args.lr

    model = torch.nn.DataParallel(model)
    model.cuda()

    print("**********************Params***********************")
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    print('params: %.2f M' % (sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6))
    print("*********************************************")

    import torch.nn as nn
    criterion = nn.L1Loss().cuda(args.gpus)

    trainable = filter(lambda x: x.requires_grad, model.parameters())

    optimizer = torch.optim.Adam(trainable, lr=init_lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150, 200, 250],
                                                     gamma=0.2)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            if args.resume_optim_sched:
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.pretrain is not None:
        from collections import OrderedDict
        checkpoint = torch.load(args.pretrain, map_location='cpu')
        loaded_state_dict = checkpoint['state_dict']
        new_state_dict = OrderedDict()
        for n, v in loaded_state_dict.items():
            name = "module." + n
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=False)

        model = model.cuda()
        print('Load pretrained weight')

    cudnn.benchmark = True
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=False, sampler=train_sampler, drop_last=True)
    val_loaders = torch.utils.data.DataLoader(
        val_datasets, batch_size=1, shuffle=False,
        num_workers=4, pin_memory=False, drop_last=False)

    print('\n\n=== Arguments ===')
    cnt = 0
    for key in sorted(vars(args)):
        print(key, ':', getattr(args, key), end='  |  ')
        cnt += 1
        if (cnt + 1) % 5 == 0:
            print('')
    print('\n')
    with open(args.save_dir + '/args.json', 'w') as args_json:
        json.dump(args.__dict__, args_json, indent=4)

    for epoch in range(args.start_epoch, args.epochs):
        train_loss, train_rmse, train_mae = train(train_loader, model, criterion, optimizer, epoch, args)
        avg_rmse = AverageMeter('avg_rmse', ':6.3f')
        val_loss, val_rmse = validate(val_loaders, model, criterion, epoch, args)
        print("{:2.3f}".format(val_rmse), end="")
        avg_rmse.update(val_rmse)

        print("Test for various Sampels/Lidars", " | RMSE {:2.3f}\n".format(avg_rmse.avg))

        total_val_rmse = avg_rmse.avg

        is_best = total_val_rmse < best_rmse
        best_rmse = min(total_val_rmse, best_rmse)

        scheduler.step()

        with open(args.save_dir + '/train.csv', 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({
                'epoch': epoch,
                'loss': train_loss,
                'rmse': train_rmse.item(),
                'mae': train_mae.item(),
                'absrel': 0,
                'delta1_02': 0,
                'delta1_05': 0,
                'delta1_10': 0,
                'delta_125_1': 0,
                'delta_125_2': 0,
                'delta_125_3': 0,
            })

        with open(args.save_dir + '/val.csv', 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({
                'epoch': epoch,
                'loss': 0,
                'rmse': total_val_rmse.item(),
                'mae': 0,
                'absrel': 0,
                'delta1_02': 0,
                'delta1_05': 0,
                'delta1_10': 0,
                'delta_125_1': 0,
                'delta_125_2': 0,
                'delta_125_3': 0,
            })
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_rmse': best_rmse,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, is_best, 'model_latest.pth.tar', args.save_dir)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    rmse = AverageMeter('RMSE', ':.4f')
    mae = AverageMeter('MAE', ':.4f')

    model.train()
    end = time.time()
    pbar = tqdm(total=len(train_loader) * args.batch_size)

    for i, sample in enumerate(train_loader):
        data_time.update(time.time() - end)

        sample = {key: val.cuda() for key, val in sample.items() if val is not None}
        args.num_images_seen += len(sample['rgb'])

        output = model(sample)
        loss1 = criterion(output['pred'], sample['gt'])

        out_rel = output['pred_init']
        out_pred = output['pred']


        maxx_out_rel = out_rel.max()
        minn_out_rel = out_rel.min()
        out_rel = (out_rel - minn_out_rel) / (maxx_out_rel - minn_out_rel)

        maxx_out_pred = out_pred.max()
        minn_out_pred = out_pred.min()
        out_pred = (out_pred - minn_out_pred) / (maxx_out_pred - minn_out_pred)

        loss_grad = criterion(kornia.filters.SpatialGradient()(out_pred), kornia.filters.SpatialGradient()(out_rel))
        
        out_rgb = output['out_rgb']
        out_dep = output['out_dep']

        L_align = F.mse_loss(out_rgb, out_dep)

        loss = loss1 + 0.1 * loss_grad + 0.05 * L_align

        losses.update(loss.item(), sample['gt'].size(0))

        rmse_result = rmse_eval(sample, output['pred'])
        rmse.update(rmse_result, sample['gt'].size(0))

        mae_result = mae_eval(sample, output['pred'])
        mae.update(mae_result, sample['gt'].size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        current_time = time.strftime('%y%m%d@%H:%M:%S')
        error_str = '{:<5s} {} | {} | Loss = {:.4f}'.format('Train', epoch, current_time, losses.avg)
        pbar.set_description(error_str)
        pbar.update(train_loader.batch_size)
    pbar.close()

    return losses.avg, rmse.avg, mae.avg


def validate(val_loader, model, criterion, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    rmse = AverageMeter('RMSE', ':.4f')

    model.eval()

    with torch.no_grad():
        for i, sample in enumerate(val_loader):
            sample = {key: val.cuda() for key, val in sample.items() if val is not None}
            end = time.time()
            output = model(sample)
            batch_time.update(time.time() - end)
            loss = criterion(output['pred'], sample['gt'])

            losses.update(loss.item(), sample['gt'].size(0))

            rmse_result = rgbdd_calc_rmse(sample, output['pred'][0, 0])
            rmse.update(rmse_result, sample['gt'].size(0))

            end = time.time()
    return losses.avg, rmse.avg


if __name__ == '__main__':
    main()
