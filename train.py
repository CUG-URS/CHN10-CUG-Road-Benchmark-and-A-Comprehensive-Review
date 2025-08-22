import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V
import cv2
import os
import numpy as np
from time import time
from networks.BSNet.BsNet import BSNet
from networks.CasNet.CasNet import CasNet
from networks.CoANet.coanet_without_HB import CoANet
from networks.DeeoRoadMapper.DeeoRoadMapper_segment import Model as DeepRoadMapper_segment
from networks.DinkNet.DinkNet import DinkNet34
from networks.DiResNet.DiResNet import FCN_Ref
from networks.GAMSNet.GAMS_Net import GAMSNet
from networks.GCBNet.GCBNet import GCBNet
from networks.ImprovedConnectivity.StackHourglassNet import StackHourglassNet
from networks.MSMDFFNet.model.MSMDFF_Net import MSMDFF_Net_v3_plus
from networks.OARENet.testNet import SwinT_OAM
from networks.roadcnn.roadcnn import Model as roadcnn
from networks.SGCN.SGCNNet import SGCN_res50
from networks.SIIS_NET.SIIS_NET import Deeplab_SIIS
from framework import MyFrame
from loss import dice_bce_loss
from data import ImageFolder


SHAPE = (512,512)
ROOT = 'dataset/train/'
imagelist = filter(lambda x: x.find('sat')!=-1, os.listdir(ROOT))
trainlist = map(lambda x: x[:-8], imagelist)
NAME = 'log01_dink34'
BATCHSIZE_PER_CARD = 32

solver = MyFrame(DinkNet34, dice_bce_loss, 2e-4)
batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD

dataset = ImageFolder(trainlist, ROOT)
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batchsize,
    shuffle=True,
    num_workers=16)

mylog = open('logs/'+NAME+'.log','w')
tic = time()
no_optim = 0
total_epoch = 300
train_epoch_best_loss = 100.
for epoch in range(1, total_epoch + 1):
    data_loader_iter = iter(data_loader)
    train_epoch_loss = 0
    for img, mask in data_loader_iter:
        solver.set_input(img, mask)
        train_loss = solver.optimize()
        train_epoch_loss += train_loss
    train_epoch_loss /= len(data_loader_iter)
    print >> mylog, '********'
    print >> mylog, 'epoch:',epoch,'    time:',int(time()-tic)
    print >> mylog, 'train_loss:',train_epoch_loss
    print >> mylog, 'SHAPE:',SHAPE
    print '********'
    print 'epoch:',epoch,'    time:',int(time()-tic)
    print 'train_loss:',train_epoch_loss
    print 'SHAPE:',SHAPE
    
    if train_epoch_loss >= train_epoch_best_loss:
        no_optim += 1
    else:
        no_optim = 0
        train_epoch_best_loss = train_epoch_loss
        solver.save('weights/'+NAME+'.th')
    if no_optim > 6:
        print >> mylog, 'early stop at %d epoch' % epoch
        print 'early stop at %d epoch' % epoch
        break
    if no_optim > 3:
        if solver.old_lr < 5e-7:
            break
        solver.load('weights/'+NAME+'.th')
        solver.update_lr(5.0, factor = True, mylog = mylog)
    mylog.flush()
    
print >> mylog, 'Finish!'
print 'Finish!'
mylog.close()