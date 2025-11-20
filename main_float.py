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
import kornia
from data.syncrgbd_dataloader_float import SyncRGBD_Dataset
from data.middlebury_dataloader_float import Middlebury_dataset
from data.nyu_dataloader_float import NYU_v2_datset
import torch.nn.functional as F
from utils.util_func import *
from utils.metric_func import *

from config import args as args_config
from model_list import import_model

warnings.filterwarnings('ignore')

args = args_config
best_rmse = 100.0
fieldnames = ['epoch', 'rmse_Sync', 'rmse_Lu', 'rmse_Midd', 'rmse_NYU', 'rmse_RGBDD', 'rmse_DIML', 'rmse_TOFDSR']


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
    best_NYU_rmse = 100.0
    best_Lu_rmse = 100.0
    best_Midd_rmse = 100.0
    args.num_images_seen = 0
    scale = args.scale

    train_dataset = SyncRGBD_Dataset(root_dir="/opt/data/share/223106110035/dataset/", txt_file='./data/SyncRGBD_Train.txt', scale=scale, train=True)#SyncRGBD_Train
    val_SyncRGBD = SyncRGBD_Dataset(root_dir="/opt/data/share/223106110035/dataset/", txt_file='./data/SyncRGBD_Test.txt', scale=scale, train=False)

    val_Lu = Middlebury_dataset(root_dir='/opt/data/private/DSR/datasets/Lu3/', scale=scale)
    val_Midd = Middlebury_dataset(root_dir='/opt/data/private/DSR/datasets/Middlebury/', scale=scale)
    val_NYU = NYU_v2_datset(root_dir="/opt/data/private/DSR/datasets/nyu_data/", scale=scale, train=False)
    test_minmax = np.load('%s/test_minmax.npy' % "/opt/data/private/DSR/datasets/nyu_data/")

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
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.decay,
                                                     gamma=0.5)  # 60, 100, 140

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
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=(train_sampler is None),
                                               num_workers=args.workers, pin_memory=False, sampler=train_sampler,
                                               drop_last=True)

    val_SyncRGBD_loaders = torch.utils.data.DataLoader(val_SyncRGBD, batch_size=1, shuffle=False, num_workers=4,
                                                    pin_memory=False, drop_last=False)
    val_Lu_loaders = torch.utils.data.DataLoader(val_Lu, batch_size=1, shuffle=False, num_workers=4, pin_memory=False,
                                                 drop_last=False)
    val_Midd_loaders = torch.utils.data.DataLoader(val_Midd, batch_size=1, shuffle=False, num_workers=4,
                                                   pin_memory=False, drop_last=False)
    val_NYU_loaders = torch.utils.data.DataLoader(val_NYU, batch_size=1, shuffle=False, num_workers=4, pin_memory=False,
                                                  drop_last=False)

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

        avg_SyncRGBD_rmse = AverageMeter('avg_rmse', ':6.3f')
        avg_Lu_rmse = AverageMeter('avg_rmse', ':6.3f')
        avg_Midd_rmse = AverageMeter('avg_rmse', ':6.3f')
        avg_NYU_rmse = AverageMeter('avg_rmse', ':6.3f')

        print("Test SyncRGBD dataset !")
        val_SyncRGBD_rmse = validate(val_SyncRGBD_loaders, model, dataname="Sync_RGBD", minmax=None)
        print("Test Lu dataset !")
        val_Lu_rmse = validate(val_Lu_loaders, model, dataname="Lu", minmax=None)
        print("Test Middlebury dataset !")
        val_Midd_rmse = validate(val_Midd_loaders, model, dataname="Midd", minmax=None)
        print("Test NYU-v2 dataset !")
        val_NYU_rmse = validate(val_NYU_loaders, model, dataname="NYU", minmax=test_minmax)

        print("{:2.3f}".format(val_SyncRGBD_rmse), end="")
        avg_SyncRGBD_rmse.update(val_SyncRGBD_rmse)
        avg_Lu_rmse.update(val_Lu_rmse)
        avg_Midd_rmse.update(val_Midd_rmse)
        avg_NYU_rmse.update(val_NYU_rmse)

        total_val_SyncRGBD_rmse = avg_SyncRGBD_rmse.avg
        total_val_Lu_rmse = avg_Lu_rmse.avg
        total_val_Midd_rmse = avg_Midd_rmse.avg
        total_val_NYU_rmse = avg_NYU_rmse.avg

        is_NYU_best = total_val_NYU_rmse < best_NYU_rmse
        best_NYU_rmse = min(total_val_NYU_rmse, best_NYU_rmse)

        is_Lu_best = total_val_Lu_rmse < best_Lu_rmse
        best_Lu_rmse = min(total_val_Lu_rmse, best_Lu_rmse)

        is_Midd_best = total_val_Midd_rmse < best_Midd_rmse
        best_Midd_rmse = min(total_val_Midd_rmse, best_Midd_rmse)

        scheduler.step()
        with open(args.save_dir + '/train.csv', 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({
                'epoch': epoch,
                'rmse_Sync': 0.0,
                'rmse_Lu': train_rmse.item(),
                'rmse_Midd': train_mae.item(),
                'rmse_NYU': 0,
                'rmse_RGBDD': 0,
                'rmse_DIML': 0,
                'rmse_TOFDSR': 0,
            })

        with open(args.save_dir + '/val.csv', 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({
                'epoch': epoch,
                'rmse_Sync': total_val_SyncRGBD_rmse.item(),
                'rmse_Lu': total_val_Lu_rmse.item(),
                'rmse_Midd': total_val_Midd_rmse.item(),
                'rmse_NYU': total_val_NYU_rmse.item(),
                'rmse_RGBDD': 0,
                'rmse_DIML': 0,
                'rmse_TOFDSR': 0,
            })
        if is_NYU_best:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_rmse': best_rmse,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, False, 'model_NYU_best.pth.tar', args.save_dir)
        if is_Lu_best:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_rmse': best_rmse,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, False, 'model_Lu_best.pth.tar', args.save_dir)
        if is_Midd_best:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_rmse': best_rmse,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, False, 'model_Midd_best.pth.tar', args.save_dir)



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

        L_rec = criterion(output['pred'], sample['gt'])

        out_rel = output['pred_init']
        out_pred = output['pred']
        maxx_out_rel = out_rel.max()
        minn_out_rel = out_rel.min()
        out_rel = (out_rel - minn_out_rel) / (maxx_out_rel - minn_out_rel)

        maxx_out_pred = out_pred.max()
        minn_out_pred = out_pred.min()
        out_pred = (out_pred - minn_out_pred) / (maxx_out_pred - minn_out_pred)
        L_grad = criterion(kornia.filters.SpatialGradient()(out_pred), kornia.filters.SpatialGradient()(out_rel))

        out_rgb = output['out_rgb']
        out_dep = output['out_dep']

        L_align = F.mse_loss(out_rgb, out_dep)

        loss = L_rec + 0.1 * L_grad + 0.05 * L_align

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


def validate(val_loader, model, dataname, minmax):
    rmse = AverageMeter('RMSE', ':.4f')

    model.eval()

    with torch.no_grad():

        for i, sample in enumerate(val_loader):
            sample = {key: val.cuda() for key, val in sample.items() if val is not None}

            output = model(sample)
            rmse_result = 0.0
            if dataname == "Sync_RGBD":
                rmse_result = syncrgbdd_calc_rmse(sample, output['pred'][0, 0])
            elif dataname == "Lu" or dataname == "Midd":
                rmse_result = midd_calc_rmse(sample, output['pred'][0, 0])
            elif dataname == "NYU":
                minmax0 = minmax[:, i]
                minmax0 = torch.from_numpy(minmax0).cuda()
                rmse_result = nyu_calc_rmse(sample, output['pred'][0, 0], minmax0)
            rmse.update(rmse_result, sample['gt'].size(0))

    return rmse.avg


if __name__ == '__main__':
    main()
