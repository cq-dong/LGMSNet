import os
import random
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

cpu_num = 1
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
import torch
torch.set_num_threads(cpu_num)
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from monai.losses import DiceCELoss, DiceLoss
from dataloader.dataset import MedicalDataSets
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from albumentations import RandomRotate90, Resize

import tempfile

from utils.util import AverageMeter
import utils.losses as losses
from utils.metrics import iou_score

from network.Tinyunet import TinyUNet
from network.CMUNeXt_new import MKtinyvit4_CT
from SOTA.ERDUnet import ERDUnet
from network.Missformer import MISSFormer

import sys
from network.IS2D_models.mfmsnet import MFMSNet
from src.network.conv_based.CMUNet import CMUNet
from src.network.conv_based.U_Net import U_Net
from src.network.conv_based.AttU_Net import AttU_Net
from src.network.conv_based.UNeXt import UNext
from src.network.conv_based.UNetplus import ResNet34UnetPlus
from src.network.conv_based.UNet3plus import UNet3plus
def get_model(args):
    if args.model == "CMUNeXt":
        model = cmunext()
    elif args.model == "CMUNeXt-S":
        model = cmunext_s()
    elif args.model == "CMUNeXt-L":
        model = cmunext_l()
    elif args.model == "TinyUNet":
        model = TinyUNet(3,1)
    elif args.model == "CMUNeXt-111":
        model = cmunext_111()
    elif args.model == "CMUNeXt_MK":
        model = CMUNeXt_MK().cuda()
    elif args.model == "U_Net":
        model = U_Net(output_ch=args.num_classes).cuda()

    elif args.model == "ERDUnet":
        model = ERDUnet().cuda()

    elif args.model == "CMUNet":
        model = CMUNet(output_ch=args.num_classes).cuda()

    elif args.model == "AttU_Net":
        model = AttU_Net(output_ch=args.num_classes).cuda()
    elif args.model == "UNext":
        model = UNext(output_ch=args.num_classes).cuda()
    elif args.model == "UNetplus":
        model = ResNet34UnetPlus(num_class=args.num_classes).cuda()
    elif args.model == "UNet3plus":
        model = UNet3plus(n_classes=args.num_classes).cuda()


   # dim[3],dim[4],dim[3],dim[2] 128,128,128,64
    elif args.model == "SegmsNet":
        model = MKtinyvit4_CT(model="CMUNeXtBlock_MK_resiual2",spilt_list=[[96,32],[96,32],[96,32],[48,16]], dims=[16, 32, 64, 128, 128]).cuda()
    elif args.model == "MKtinyvit4_CT_v66_MK_resiual2_1_a":
        model = MKtinyvit4_CT(model="CMUNeXtBlock_MK_resiual2",spilt_list=[[96,32],[96,32],[96,32],[48,16]], dims=[16, 32, 64, 128, 128]).cuda()

    
    else:
        print("123")
        print("model err:",args.model)
        # model = None
        exit(0)
    return model.cuda()


def seed_torch(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # GPU随机种子确定
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.use_deterministic_algorithms(True,warn_only=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="CMUNeXt", help='model')
parser.add_argument('--base_dir', type=str, default="./data/busi", help='dir')
parser.add_argument('--train_file_dir', type=str, default="busi_train.txt", help='dir')
parser.add_argument('--val_file_dir', type=str, default="busi_val.txt", help='dir')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--batch_size', type=int, default=8,
                    help='batch_size per gpu')
parser.add_argument('--gpu', type=str, default="7", help='gpu')
parser.add_argument('--epoch', type=int, default=300, help='epoch')
parser.add_argument('--seed', type=int, default=41, help='seed')
parser.add_argument('--txtnum', type=int, default=1, help='txtnum')
parser.add_argument('--img_size', type=int, default=256, help='img_size')
parser.add_argument('--num_classes', type=int, default=1, help='img_size')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
print(os.environ["CUDA_VISIBLE_DEVICES"] )
seed_torch(args.seed)



def getDataloader():
    img_size = args.img_size
    train_transform = Compose([
        RandomRotate90(),
        transforms.Flip(),
        Resize(img_size, img_size),
        transforms.Normalize(),
    ])

    val_transform = Compose([
        Resize(img_size, img_size),
        transforms.Normalize(),
    ])
    db_train = MedicalDataSets(base_dir=args.base_dir, split="train", transform=train_transform,
                               train_file_dir=args.train_file_dir, val_file_dir=args.val_file_dir)
    db_val = MedicalDataSets(base_dir=args.base_dir, split="val", transform=val_transform,
                             train_file_dir=args.train_file_dir, val_file_dir=args.val_file_dir)
    print("train num:{}, val num:{}".format(len(db_train), len(db_val)))

    trainloader = DataLoader(db_train, batch_size=args.batch_size, shuffle=True,
                             num_workers=4, pin_memory=False)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)
    return trainloader, valloader

def train(args):
    base_lr = args.base_lr
    trainloader, valloader = getDataloader()
    model = get_model(args)
    model_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("train file dir:{} val file dir:{}".format(args.train_file_dir, args.val_file_dir))
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    criterion = losses.__dict__['BCEDiceLoss']().cuda()
    print("{} iterations per epoch".format(len(trainloader)))
    best_iou = 0
    best_epoch=0
    best_iou_withSE=0
    best_iou_withPC=0
    best_iou_withF1=0
    best_iou_withACC=0
    iter_num = 0
    max_epoch = args.epoch
    max_iterations = len(trainloader) * max_epoch

    train_loss_list=[]
    train_iou_list=[]
    loss_list=[]
    iou_list=[]
    f1_list=[]
    val_save_path_list=None
    for epoch_num in tqdm(range(max_epoch), desc='Training Progress'):
        model.train()
        avg_meters = {'loss': AverageMeter(),
                      'iou': AverageMeter(),
                      'val_loss': AverageMeter(),
                      'val_iou': AverageMeter(),
                      'SE': AverageMeter(),
                      'PC': AverageMeter(),
                      'F1': AverageMeter(),
                      'ACC': AverageMeter()
                      }
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            outputs = model(volume_batch)
            
            loss = criterion(outputs, label_batch)
            iou, dice, _, _, _, _, _ = iou_score(outputs, label_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            avg_meters['loss'].update(loss.item(), volume_batch.size(0))
            avg_meters['iou'].update(iou, volume_batch.size(0))

        model.eval()
        with torch.no_grad():
            for i_batch, sampled_batch in enumerate(valloader):
                input, target = sampled_batch['image'], sampled_batch['label']
                input = input.cuda()
                target = target.cuda()
                output = model(input)
                loss = criterion(output, target)
                
                iou, _, SE, PC, F1, _, ACC = iou_score(output, target)
                avg_meters['val_loss'].update(loss.item(), input.size(0))
                avg_meters['val_iou'].update(iou, input.size(0))
                avg_meters['SE'].update(SE, input.size(0))
                avg_meters['PC'].update(PC, input.size(0))
                avg_meters['F1'].update(F1, input.size(0))
                avg_meters['ACC'].update(ACC, input.size(0))


        train_loss_list.append(avg_meters['loss'].avg)
        loss_list.append(avg_meters['val_loss'].avg)
        iou_list.append(avg_meters['val_iou'].avg)
        f1_list.append(avg_meters['F1'].avg)


        print(
            'epoch [%d/%d]  train_loss : %.4f, train_iou: %.4f '
            '- val_loss %.4f - val_iou %.4f - val_SE %.4f - val_PC %.4f - val_F1 %.4f - val_ACC %.4f'
            % (epoch_num, max_epoch, avg_meters['loss'].avg, avg_meters['iou'].avg,
               avg_meters['val_loss'].avg, avg_meters['val_iou'].avg, avg_meters['SE'].avg,
               avg_meters['PC'].avg, avg_meters['F1'].avg, avg_meters['ACC'].avg))

        if avg_meters['val_iou'].avg > best_iou:
            # if not os.path.exists('./checkpoint'):
            #     os.mkdir('checkpoint')
            # torch.save(model.state_dict(), 'checkpoint/{}_model_{}.pth'
            #            .format(args.model, args.train_file_dir.split(".")[0]))
            best_iou = avg_meters['val_iou'].avg
            if not os.path.exists(f'./checkpoint/{args.model}/'):
                os.mkdir(f'./checkpoint/{args.model}/')
            tem_save_path=f'./checkpoint/{args.model}/{args.model}_model_{args.train_file_dir.split(".")[0]}_{args.seed}_valiou_{best_iou:.4f}.pth'
            torch.save(model.state_dict(), tem_save_path)
            if val_save_path_list:
                if os.path.exists(val_save_path_list):
                    # 删除文件
                    os.remove(val_save_path_list)
            val_save_path_list= tem_save_path 
            '''
            torch.save(model.state_dict(), 'checkpoint/{}_model_{}_{}.pth'
                       .format(args.model, args.train_file_dir.split(".")[0]))
            '''
            best_epoch = epoch_num
            best_iou_withF1 = avg_meters['F1'].avg
            best_iou_withSE = avg_meters['SE'].avg
            best_iou_withPC = avg_meters['PC'].avg
            best_iou_withACC= avg_meters['ACC'].avg

            print("=> saved best model")
            
    
    
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    epochs = list(range(len(train_loss_list)))
    # Plot training loss
    axs[0, 0].plot(train_loss_list)
    # axs[0, 0].scatter(epochs, train_loss_list)
    axs[0, 0].set_title('Training Loss')
    axs[0, 0].set_xlabel('Epoch')
    # Plot validation loss
    axs[0, 1].plot(loss_list)
    # axs[0, 1].scatter(epochs, loss_list)
    axs[0, 1].set_title('Validation Loss')
    axs[0, 0].set_xlabel('Epoch')
    # Plot validation IoU
    axs[1, 0].plot(iou_list)
    # axs[1, 0].scatter(epochs, iou_list)
    axs[1, 0].set_title('Validation IoU')
    axs[0, 0].set_xlabel('Epoch')
    # Plot validation F1
    axs[1, 1].plot(f1_list)
    # axs[1, 1].scatter(epochs, f1_list)
    axs[1, 1].set_title('Validation F1')
    axs[0, 0].set_xlabel('Epoch')
    plt.tight_layout()
    # Save the figure
    plt.savefig(f'./outputpng/{args.model}_{args.train_file_dir}_{args.batch_size}_{args.epoch}_{args.seed}_{args.base_lr}_{best_iou:.4f}.png')

    
    return best_iou,best_epoch,model_parameters,[best_iou_withF1,best_iou_withSE ,best_iou_withPC ,best_iou_withACC]


if __name__ == "__main__":
    with open(f"./result{args.txtnum}.txt", 'a') as f:
        f.write(f"\nModel:{args.model.ljust(20)} ")
        best_iou,best_epoch,model_parameters,others=train(args)
        f.write(f" best_epoch:{best_epoch} best_iou:{best_iou:.6f} model_parameters:{model_parameters} train_file_dir:{args.train_file_dir} val_file_dir:{args.val_file_dir}  batch_size:{args.batch_size} epoch:{args.epoch} seed:{args.seed} base_lr:{args.base_lr} imgsize:{args.img_size} F1:{others[0]} SE:{others[1]} PC:{others[2]} ACC:{others[3]}")
