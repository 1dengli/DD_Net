import pandas as pd
import argparse
import os
from collections import OrderedDict
import yaml
from load_LIDC_data import LIDC_IDRI
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import albumentations as albu
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from losses import BCEDiceLoss
from metrics1 import iou_score,dice_coef
from utils import AverageMeter, str2bool
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
import random
import numpy as np
import matplotlib.pyplot as plt

#网络模型
from unety import UNett_batcnnorm
from unety.SAR_UNet import Se_PPP_ResUNet
from Swin_Unet import vision_transformer
from TransUNet.networks.vit_seg_modeling import VisionTransformer as ViT_seg
from TransUNet.networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from unety import attention_unet
from ResUNet_family import res_unet_plus
from UNet2P import UNet_2Plus
from UNet3P.models import UNet_3Plus
import resunet_aspp_up_vit4
from UCTransNet.nets import UCTransNet
from UACANet_main.lib.UACANet import UACANet
import DDANet
import fcn
from MSNet_M2SNet_main.model.miccai_msnet import M2SNet,MSNet


#my model

from Res2Net_U.ResNet_mynet import RESNet
from Res2Net_U.ResNet_mynet_no_F import ResNet_mynet_no_F
from Res2Net_U.ResNet_mynet_boundary import ResNet_mynet_boundary
from Res2Net_U.Res2Net_mynet_boundary import Res2Net_mynet_boundary
from Res2Net_U.ResNet_mynet_boundary_TAFF import ResNet_mynet_boundary_TAFF
from Res2Net_U.ResNet_mynet_boundary_TAFF_CDC import ResNet_mynet_boundary_TAFF_CDC
from Res2Net_U.ResNet_mynet_boundary_TAFF_CDC_no_eage import ResNet_mynet_boundary_TAFF_CDC_no_edge
from Res2Net_U.ResNet_mynet_boundary_TAFF_no_CDC import ResNet_mynet_boundary_TAFF_no_CDC
# from ce2 import ce2
# from ce3 import ce3
# from ce4 import ce4
# from ce5 import ce5
# from ce6 import ce6
# from ce7 import ce7
# from ce8 import ce8
# from ce9 import ce9
# from ce10 import ce10
# from ce11 import ce11
# from ce12 import ce12
# from ce13 import ce13
# from x1 import x1
# from x2 import x2
# from x10 import x10
# from x12 import x12
from m1 import m1
from m4 import m4
from m6 import m6
from m11 import m11
from m12 import m12
from m13 import m13
from m14 import m14
from m15 import m15
from m16 import m16
from m17 import m17
from m18 import m18
from m19 import m19
from m36 import m36
from m40 import m40
from m43 import m43
from m57 import m57
from m37 import m37
from a98 import a98
from a101010 import a101010
from m000 import m000
from DA_TransUnet_main.DA_TransUNet.Architecture import DATransUNet as DA_TransUNet
from DA_TransUnet_main.DA_TransUNet.Architecture import DATransUNet
from BRAU_Netplusplus_Pytorch_main.models import bra_unet
import time

def parse_args():
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--name', default="DD_Net",
                        help='model name: UNET', choices=['UNET', 'SAR_UNet', 'UNet2P','UNet3P', 'swinunet', 'UCTransNet',"resunetpp"
                                                         'attention_unet', 'TransUNet',"DDANet","M2SNet","Res2Net_UNet","RESNet","ResNet_mynet_boundary",
                                                         "Res2Net_mynet_boundary","ResNet_mynet_boundary_TAFF","ResNet_mynet_boundary_TAFF_CDC",
                                                         "ResNet_mynet_boundary_TAFF_CDC_no_edge","ResNet_mynet_boundary_TAFF_no_CDC","ResNet_mynet_no_F"])
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=8, type=int,
                        metavar='N', help='mini-batch size (default: 6)')
    parser.add_argument('--early_stopping', default=50, type=int,
                        metavar='N', help='early stopping (default: 50)')
    parser.add_argument('--num_workers', default=8, type=int)
    # optimizer
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                        ' | '.join(['Adam', 'SGD']) +
                        ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=5e-5, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')
    # data
    parser.add_argument('--augmentation',type=str2bool,default=False,choices=[True,False])
    config = parser.parse_args()

    return config


def train(train_loader, model, criterion,  optimizer):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter()}
    model.train()
    pbar = tqdm(total=len(train_loader))
    for input, target, _, _ in train_loader:
        # b,c,h,w = input.size()
        # input = input.reshape(b,c,h,w)
        input = input.cuda()
        target = target.cuda()
        # canny = canny.cuda()

        output = model(input)
        # output2 = torch.squeeze(output2)
        output = torch.squeeze(output)
        target = torch.squeeze(target)
        loss = criterion(output, target)
        # loss2 = criterion2(output2, canny)

        iou = iou_score(output, target)
        dice = dice_coef(output, target)

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))
        avg_meters['dice'].update(dice, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
            ('dice',avg_meters['dice'].avg)
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice',avg_meters['dice'].avg)
                        ])

def validate(val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter()}
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _, _ in val_loader:
            # b, h, w, c = input.size()
            # input = input.reshape(b, c, h, w)
            input = input.cuda()
            target = target.cuda()
            output = model(input)
            output = torch.squeeze(output)
            target = torch.squeeze(target)
            loss = criterion(output, target)
            iou = iou_score(output, target)
            dice = dice_coef(output, target)
            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice',avg_meters['dice'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice',avg_meters['dice'].avg)])

def main():
    deterministic = True
    if not deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(2023)
    np.random.seed(2023)
    torch.manual_seed(2023)
    torch.cuda.manual_seed(2023)
    torch.cuda.manual_seed_all(2023)
    # Get configuration
    config = vars(parse_args())
    # Make Model output directory
    if config['augmentation']== True:
        file_name= config['name'] + '_with_augmentation'
    else:
        file_name = config['name'] +'_base'
    os.makedirs('p3/fjj/{}'.format(file_name),exist_ok=True)
    print("Creating directory called", file_name)

    print('-' * 20)
    print("Configuration Setting as follow")
    for key in config:
        print('{}: {}'.format(key, config[key]))
    print('-' * 20)

    #save configuration
    with open('p3/fjj/{}/config.yml'.format(file_name), 'w') as f:
        yaml.dump(config, f)

    #criterion = nn.BCEWithLogitsLoss().cuda()
    criterion = BCEDiceLoss().cuda()
    # criterion2 = nn.BCEWithLogitsLoss().cuda()

    # create model
    print("=> creating model")
    if config['name'] == "UNET":
        model = UNett_batcnnorm.Unet(1, 1)
    elif config['name'] == "swinunet":
        model = vision_transformer.SwinUnet(img_size=128, num_classes=1)
    elif config['name'] == "UNet3P":
        model = UNet_3Plus.UNet_3Plus(in_channels=1, n_classes=1)
    elif config['name'] == "UNet2P":
        model = UNet_2Plus.Unet2_Plus(num_classes=1)
    elif config['name'] == 'SAR_UNet':
        model = Se_PPP_ResUNet(1, 1, deep_supervision=False)
    elif config['name'] == 'TransUNet':
        config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
        model = ViT_seg(config_vit, img_size=128, num_classes=1)
    elif config['name'] == 'attention_unet':
        model = attention_unet.AttU_Net(1, 1)
    elif config['name'] == "resunetpp":
        model = res_unet_plus.ResUnetPlusPlus(1, 1)
    elif config['name'] == "UCTransNet":
        model = UCTransNet.UCTransNet(1, 1,img_size=128)
    elif config['name'] == "DDANet":
        model = DDANet.CompNet()
    elif config['name'] == "M2SNet":
        model = M2SNet()
    elif config['name'] == "MSNet":
        model = MSNet()
    elif config['name'] == "RESNet":
        model = RESNet(1)
    elif config['name'] == "m40":
        model = m40(1)
    elif config['name'] == "m37":
        model = m37(1)
    elif config['name'] == "m36":
        model = m36(1)
    elif config['name'] == "DD_Net":
        model = DDNet(1)
    elif config['name'] == "m17":
        model = m17(1)
    elif config['name'] == "m43":
        model = m43(1)
    elif config['name'] == "a101010":
        model = a101010(1)
    # elif config['name'] == "ce2":
    #     model = ce2(1)
    # elif config['name'] == "ce3":
    #     model = ce3(1)
    # elif config['name'] == "ce4":
    #     model = ce4(1)
    # elif config['name'] == "ce5":
    #     model = ce5(1)
    # elif config['name'] == "ce6":
    #     model = ce6(1)
    # elif config['name'] == "ce7":
    #     model =ce7(1)
    # elif config['name'] == "ce8":
    #     model = ce8(1)
    # elif config['name'] == "ce9":
    #     model = ce9(1)
    # elif config['name'] == "ce10":
    #     model = ce10(1)
    # elif config['name'] == "ce11":
    #     model = ce11(1)
    # elif config['name'] == "ce12":
    #     model = ce12(1)
    # elif config['name'] == "ce13":
    #     model = ce13(1)
    # elif config['name'] == "x1":
    #     model = x1(1)
    # elif config['name'] == "x2":
    #     model = x2(1)
    # elif config['name'] == "x10":
    #     model = x10(1)
    elif config['name'] == "m4":
        model = m4(num_classes=1, encoder='pvt_v2_b2', pretrain=True, pretrained_dir='E:\\fjj2\\lib\\')
    elif config['name'] == "DA_TransUnet":
        config_da = DATransUNet.CONFIGS['R50-ViT-B_16']
        model = DA_TransUNet.DA_Transformer(config_da, 128, 1)
    elif config['name'] == "BRAU_Net++":
        model = bra_unet.BRAUnet(img_size=128, in_chans=1, n_win=4)
    else:
        raise ValueError("Wrong Parameters")
    model = model.cuda()

    torch.cuda.reset_peak_memory_stats()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    params = filter(lambda p: p.requires_grad, model.parameters())
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'], nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError


    # exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    dataset = LIDC_IDRI(dataset_location='D:\\fjj_code\\resnet_seg\\data\\')
    train_sampler = torch.load('sampler/bingji/train_sampler.pth')
    val_sampler = torch.load('sampler/bingji/val_sampler.pth')
    train_loader = DataLoader(dataset, batch_size=config['batch_size'], sampler=train_sampler, shuffle=False, drop_last=True)
    val_loader = DataLoader(dataset, batch_size=config['batch_size'], sampler=val_sampler, shuffle=False, drop_last=True)

    log = pd.DataFrame(
        columns=['epoch', 'lr', 'loss', 'iou', 'dice', 'val_loss', 'val_iou', 'val_dice']
    )

    best_dice = 0
    trigger = 0
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=5e-6)
    start_time = time.time()

    for epoch in range(config['epochs']):
        train_log = train(train_loader, model, criterion, optimizer)
        val_log = validate(val_loader, model, criterion)
        scheduler.step()  # 更新学习率

        print(
            'Training epoch [{}/{}], Training BCE loss:{:.4f}, Training DICE:{:.4f}, Training IOU:{:.4f}, Validation BCE loss:{:.4f}, Validation Dice:{:.4f}, Validation IOU:{:.4f}'.format(
                epoch + 1, config['epochs'], train_log['loss'], train_log['dice'], train_log['iou'], val_log['loss'],
                val_log['dice'], val_log['iou']))

        # 1) 先创建 Series
        tmp = pd.Series([
            epoch,
            config['lr'],
            train_log['loss'],
            train_log['iou'],
            train_log['dice'],
            val_log['loss'],
            val_log['iou'],
            val_log['dice'],
        ], index=log.columns)

        # 2) 把它变成一个 1×8 的 DataFrame
        new_row = tmp.to_frame().T

        # 3) 再 concat，就能按行加了
        log = pd.concat([log, new_row], ignore_index=True)
        log.to_csv('p3/fjj/{}/log.csv'.format(file_name), index=False)

        trigger += 1

        if val_log['dice'] > best_dice:
            torch.save(model.state_dict(), 'p3/fjj/{}/bestmodel_{}_CosineAnnealingLR.pth'.format(file_name,config["lr"]))
            best_dice = val_log['dice']
            print("=> saved best model as validation DICE is greater than previous best DICE")
            trigger = 0

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()

        peak_memory = torch.cuda.max_memory_allocated() / 1024 ** 2
        print("Peak GPU memory (MB):", peak_memory)

        total_time = time.time() - start_time
        print("Total training time (hours):", total_time / 3600)

if __name__ == '__main__':
    main()