import os
import time
import torch
import utils
import config
import torchvision
import torch.nn as nn
import numpy as np
import random
from model_snn import SNASNet, find_best_neuroncell
from utils import data_transforms
from spikingjelly.clock_driven.functional import reset_net


def main():
    args = config.get_args()

    # define dataset
    #-------------------------------------------------------------------------
    # 加载实验数据集
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.Grayscale(),# 转成单通道的灰度图
        # 把值转成Tensor
        torchvision.transforms.ToTensor()])

    #dataset = torchvision.datasets.ImageFolder("/kaggle/input/ddos-2019/Dataset-4/Dataset-4", 
    #                                            transform=transform)
    dataset = torchvision.datasets.ImageFolder("/kaggle/input/cic-2018-for-snn", 
                                                transform=transform)
    #dataset = torchvision.datasets.ImageFolder("/kaggle/input/nsl-kdd-for-snn/data", 
    #                                            transform=transform)

    # 切分，训练集和验证集
    random.seed(0)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    split_point = int(0.8*len(indices))
    train_indices = indices[:split_point]
    test_indices = indices[split_point:]

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                          sampler=torch.utils.data.SubsetRandomSampler(train_indices))
    # 用于探索最佳网络结构
    trainset = train_loader.dataset

    val_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                         sampler=torch.utils.data.SubsetRandomSampler(test_indices))

    #-------------------------------------------------------------------------


    if args.cnt_mat is None: # serach neuroncell if no predefined neuroncell
        best_neuroncell = find_best_neuroncell(args, trainset)
    else:
        int_list = []
        for line in args.cnt_mat:
            row_list = []
            for element in line:
                row_list.append(int(element))
            int_list.append(row_list)
        best_neuroncell = torch.Tensor(int_list)




    print ('-'*7, "best_neuroncell",'-'*7)
    print (best_neuroncell)
    print('-' * 30)

    # Reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


    model = SNASNet(args, best_neuroncell).cuda()
    criterion = nn.CrossEntropyLoss().cuda()

    # 设置优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 自动调整学习率
    # 余弦退火, T_max是cos周期1/4（函数值从1到0需要经过的迭代周期）
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                    T_max=args.epochs, verbose=True)


    start = time.time()
    for epoch in range(args.epochs):
        train(args, epoch, train_loader, model, criterion, optimizer, scheduler)
        scheduler.step()
        if (epoch + 1) % 3 == 0:
            validate(args, epoch, val_loader, model, criterion)
            #utils.save_checkpoint({'state_dict': model.state_dict(), }, epoch + 1, tag=args.exp_name + '_super')
            # 保存模型训练结果
            torch.save(model, '/kaggle/working/trained-model'+str(epoch+1)+'.pt')
    utils.time_record(start)
    # 保存模型训练结果
    torch.save(model, '/kaggle/working/trained-CIC.pt')


def train(args, epoch, train_data,  model, criterion, optimizer, scheduler):
    model.train()
    train_loss = 0.0
    top1 = utils.AvgrageMeter()
    if (epoch + 1) % 10 == 0:
        print('[%s%04d/%04d %s%f]' % ('Epoch:', epoch + 1, args.epochs, 'lr:', scheduler.get_lr()[0]), flush=True)

    for step, (inputs, targets) in enumerate(train_data):
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 5))
        n = inputs.size(0)
        top1.update(prec1.item(), n)
        train_loss += loss.item()
        reset_net(model)
        
    print('train_loss: %.6f' % (train_loss / len(train_data)), 'train_acc: %.6f' % top1.avg, flush=True)

def validate(args, epoch, val_data, model, criterion):
    model.eval()
    val_loss = 0.0
    val_top1 = utils.AvgrageMeter()

    with torch.no_grad():
        for step, (inputs, targets) in enumerate(val_data):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 5))
            n = inputs.size(0)
            val_top1.update(prec1.item(), n)
            reset_net(model)
        print('[Val_Accuracy epoch:%d] val_acc:%f'
              % (epoch + 1,  val_top1.avg), flush=True)
        return val_top1.avg


if __name__ == '__main__':
    main()
