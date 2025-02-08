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
# from dataloader.dataset import MedicalDataSets
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from albumentations import RandomRotate90, Resize
from datamodule_my import KvasirSEGDataset

from utils.util import AverageMeter
import utils.losses as losses
from utils.metrics import iou_score
from network.CMUNeXt_new import MKtinyvit4_CT


from src.network.conv_based.CMUNet import CMUNet
from src.network.conv_based.U_Net import U_Net
from src.network.conv_based.AttU_Net import AttU_Net
from src.network.conv_based.UNeXt import UNext
from src.network.conv_based.UNetplus import ResNet34UnetPlus
from src.network.conv_based.UNet3plus import UNet3plus
from src.network.conv_based.CMUNeXt import cmunext

from src.network.transfomer_based.transformer_based_network import get_transformer_based_model
from network_A.CMUNeXt_new import MKtinyvit4_CT_Ablation1,MKtinyvit4_CT_Ablationb,MKtinyvit4_CT_S1
from network.Missformer import MISSFormer
from network.UniRepLKNet import Utr

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
parser.add_argument('--epoch', type=int, default=1, help='epoch')
parser.add_argument('--seed', type=int, default=41, help='seed')
parser.add_argument('--txtnum', type=int, default=1, help='txtnum')
parser.add_argument('--img_size', type=int, default=256, help='img_size')
parser.add_argument('--num_classes', type=int, default=1, help='img_size')

args = parser.parse_args()
# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
print(os.environ["CUDA_VISIBLE_DEVICES"] )
seed_torch(args.seed)


          def getDataloader():
    img_size = args.img_size

    dataset = KvasirSEGDataset(batch_size=args.batch_size, img_size=img_size)
    dataset.setup()
    train_set,val_set,test_set=dataset.train_set,dataset.val_set,dataset.test_set
    train_dataloader =  DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
        )
    val_dataloader =  DataLoader(
            val_set,
            batch_size=1,
            shuffle=False,
            num_workers=4,
        )
    test_dataloader =  DataLoader(
            test_set,
            batch_size=1,
            shuffle=False,
            num_workers=4,
        )
    # return dataset.train_dataloader(), dataset.val_dataloader(),dataset.test_dataloader()
    return train_dataloader,val_dataloader,test_dataloader

def train(args):
    base_lr = args.base_lr
    trainloader, valloader,test_dataloader = getDataloader()
    model = get_model(args)
    model_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("train file dir:{} val file dir:{}".format(args.train_file_dir, args.val_file_dir))
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    criterion = losses.__dict__['BCEDiceLoss']().cuda()
    print("{} iterations per epoch".format(len(trainloader)))
    best_iou = 0
    best_test_iou=0
    best_iou_withf1=0
    best_val_metric = {
        "epoch":0,
        'val_iou': 0,
        'val_SE': 0,
        'val_PC':0,
        'val_F1': 0,
        'val_ACC': 0,
    }
    best_val_withtest_metric = {
        "epoch":0,
        'test_iou': 0,
        'test_SE': 0,
        'test_PC':0,
        'test_F1': 0,
        'test_ACC': 0,
    }
    best_test_metric = {
        "epoch":0,
        'test_iou': 0,
        'test_SE': 0,
        'test_PC':0,
        'test_F1': 0,
        'test_ACC': 0,
    }
    iter_num = 0
    max_epoch = args.epoch
    max_iterations = len(trainloader) * max_epoch

    train_loss_list=[]
    train_iou_list=[]
    loss_list=[]
    iou_list=[]
    f1_list=[]

    val_save_path_list=None
    test_save_path_list=None

    for epoch_num in tqdm(range(max_epoch), desc='Training Progress'):
        model.train()
        avg_meters = {'loss': AverageMeter(),
                      'iou': AverageMeter(),
                      'val_loss': AverageMeter(),
                      'val_iou': AverageMeter(),
                      'val_SE': AverageMeter(),
                      'val_PC': AverageMeter(),
                      'val_F1': AverageMeter(),
                      'val_ACC': AverageMeter(),
                      'test_iou': AverageMeter(),
                      'test_SE': AverageMeter(),
                      'test_PC': AverageMeter(),
                      'test_F1': AverageMeter(),
                      'test_ACC': AverageMeter()
                      }
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            outputs = model(volume_batch)
            # print(outputs.shape,label_batch.shape)
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
                avg_meters['val_SE'].update(SE, input.size(0))
                avg_meters['val_PC'].update(PC, input.size(0))
                avg_meters['val_F1'].update(F1, input.size(0))
                avg_meters['val_ACC'].update(ACC, input.size(0))
        train_loss_list.append(avg_meters['loss'].avg)
        loss_list.append(avg_meters['val_loss'].avg)
        iou_list.append(avg_meters['val_iou'].avg)
        f1_list.append(avg_meters['val_F1'].avg)


        with torch.no_grad():
            for i_batch, sampled_batch in enumerate(test_dataloader):
                input, target = sampled_batch['image'], sampled_batch['label']
                input = input.cuda()
                target = target.cuda()
                output = model(input)
                loss = criterion(output, target)
                
                iou, _, SE, PC, F1, _, ACC = iou_score(output, target)
                avg_meters['test_iou'].update(iou, input.size(0))
                avg_meters['test_SE'].update(SE, input.size(0))
                avg_meters['test_PC'].update(PC, input.size(0))
                avg_meters['test_F1'].update(F1, input.size(0))
                avg_meters['test_ACC'].update(ACC, input.size(0))

        print(
            'epoch [%d/%d]  train_loss : %.4f, train_iou: %.4f '
            '- val_loss %.4f - val_iou %.4f - val_SE %.4f - val_PC %.4f - val_F1 %.4f - val_ACC %.4f'
            % (epoch_num, max_epoch, avg_meters['loss'].avg, avg_meters['iou'].avg,
               avg_meters['val_loss'].avg, avg_meters['val_iou'].avg, avg_meters['val_SE'].avg,
               avg_meters['val_PC'].avg, avg_meters['val_F1'].avg, avg_meters['val_ACC'].avg))

        if avg_meters['val_iou'].avg > best_iou:
            best_iou = avg_meters['val_iou'].avg
            if not os.path.exists(f'./checkpoint/{args.model}'):
                os.mkdir(f'./checkpoint/{args.model}')
            tem_save_path=f'./checkpoint/{args.model}/{args.model}_model_{args.train_file_dir.split(".")[0]}_{args.seed}_valiou_{best_iou:.4f}.pth'
            torch.save(model.state_dict(), tem_save_path)
            if val_save_path_list:
                if os.path.exists(val_save_path_list):
                    # 删除文件
                    os.remove(val_save_path_list)
            val_save_path_list= tem_save_path       

            best_iou_withf1 = avg_meters['val_F1'].avg
            for key in best_val_metric:
                if key=="epoch":
                    best_val_metric[key]=epoch_num
                else:
                    best_val_metric[key]=avg_meters[key].avg
            for key in best_val_withtest_metric:
                if key=="epoch":
                    best_val_metric[key]=epoch_num
                else:
                    best_val_withtest_metric[key]=avg_meters[key].avg
        if  avg_meters['test_iou'].avg > best_test_iou:
            best_test_iou=avg_meters['test_iou'].avg
            if not os.path.exists(f'./checkpoint/{args.model}'):
                os.mkdir(f'./checkpoint/{args.model}')
            # torch.save(model.state_dict(), f'./checkpoint/{args.model}/{args.model}_model_{args.train_file_dir.split(".")[0]}_{args.seed}_testiou_{avg_meters['test_iou'].avg:.4f}.pth')
            tem_save_path=f'./checkpoint/{args.model}/{args.model}_model_{args.train_file_dir.split(".")[0]}_{args.seed}_testiou_{best_test_iou:.4f}.pth'
            torch.save(model.state_dict(), tem_save_path)
            if test_save_path_list:
                if os.path.exists(test_save_path_list):
                    # 删除文件
                    os.remove(test_save_path_list)    
            test_save_path_list= tem_save_path       
            for key in best_val_withtest_metric:
                if key=="epoch":
                    best_val_metric[key]=epoch_num
                else:
                    best_test_metric[key]=avg_meters[key].avg
            print("=> saved test best model")
    
    
    
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

    
    return best_iou,best_iou_withf1,model_parameters,[best_val_metric,best_val_withtest_metric,best_test_metric]


if __name__ == "__main__":
    with open(f"./result{args.txtnum}.txt", 'a') as f:
        f.write(f"\nModel:{args.model.ljust(20)} ")
        best_iou,best_iou_withf1,model_parameters,dictlist=train(args)
        f.write(f" best_iou:{best_iou:.6f} best_iou_withf1:{best_iou_withf1:.6f} model_parameters:{model_parameters} train_file_dir:{args.train_file_dir} val_file_dir:{args.val_file_dir}  batch_size:{args.batch_size} epoch:{args.epoch} seed:{args.seed} base_lr:{args.base_lr} imgsize:{args.img_size}")
        for dic in dictlist:
            # 遍历字典中的每个键值对
            f.write("\n")
            for key, value in dic.items():
                # 将键值对写入文件，每个键值对占一行，值保留6位小数，并在键和值之间加入制表符
                f.write(f'{key}\t{value:.6f}\t')