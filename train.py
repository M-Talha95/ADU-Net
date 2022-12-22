from __future__ import print_function
import torch
import torch.nn as nn
#import logging
import torch.backends.cudnn as cudnn
import time
import torch.nn.functional as F
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import logging
from metric import SegmentationMetric
from measure import SegmentationMetric_foreground
from dataset import train_dataset
#from MACUNet import MACUNet
from early_stopping import EarlyStopping
from tqdm import tqdm, trange
from models.UNet import UNet, UNetLAM, UNetLinear, UNetDense, DUNet, DUNetDense, DUNetLAM, DUNetLAMCM, DUNetDenseLAM, PUNet, DUNetDenseLAMfour, UNetBilinear, DUNetDenseLinearAttention
import color_map_to_class_index #import apply_GID_colormap
#from UNet_Dense_functions import UNetOriginal
#from sklearn.metrics import precision_recall_fscore_support as score
from classcount import classcount



batch_size = 8
#batch_size=32 ###for debugging purpose a
niter = 100
class_num = 5
learning_rate = 0.0001
beta1 = 0.5
cuda = True
num_workers = 0
size_h = 256
size_w = 256
flip = 0
band = 3
#net = MACUNet(band, class_num)
#net =DUNetDenseLAMfour(band, class_num)
net = UNetBilinear(band, class_num)
train_path = r'D:\Study\Github\land_cover_classification_unet\Dataset\train/'
val_path = r'D:\Study\Github\land_cover_classification_unet\Dataset\val/'
# test_path = r'E:\Talha\3\Talha\GID_5\test/'
test_path = r'E:\Talha\3\Talha\GID_5\testing\test/'
out_file = r'D:\Study\Github\land_cover_classification_unet\checkpoint/' + net.name
#out_file = r'E:\Talha\3\Talha\GID_5/' + net.name

save_epoch = 1
test_step = 300
log_step = 1
num_GPU = 1
index = 2000
pre_trained = True
#torch.cuda.set_device(0)

try:
    import os
    os.makedirs(out_file)
except OSError:
    pass

manual_seed = 10
random.seed(manual_seed)
torch.manual_seed(manual_seed)
cudnn.benchmark = True

train_datatset_ = train_dataset(train_path, size_w, size_h, flip, band, batch_size)
val_datatset_ = train_dataset(val_path, size_w, size_h, 0, band)

def weights_init(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif class_name.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

try:
    os.makedirs(out_file)
    os.makedirs(out_file + '/')
except OSError:
    pass
# if cuda:
#     net.cuda()
# if num_GPU > 1:
#     net = nn.DataParallel(net)

if pre_trained and os.path.exists('%s/' % out_file + 'netG.pth'):
    net.load_state_dict(torch.load('%s/' % out_file + 'netG.pth'))
    print('Load success!')
else:
    pass
    # net.apply(weights_init)

###########   LOSS & OPTIMIZER   ##########
criterion = nn.CrossEntropyLoss(ignore_index=255)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, eps=1e-08)
#optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.90, weight_decay=0.0005)
metric = SegmentationMetric(class_num)
#measure = SegmentationMetric_foreground(class_num)
early_stopping = EarlyStopping(patience=30, verbose=True)

if __name__ == '__main__':
    start = time.time()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net = net.to(device)
    losses=[];mean_losses=[]
    iter_ =0
    # Allpred=[]
    # Allgt=[]
    
    weights_classes = classcount(train_datatset_)
    weights_classes = weights_classes.to(device=device, dtype=torch.float32)

    print("Class Count of Training Set", weights_classes)
   
    net.train()
    # global train_seg_iou
    # train_seg_loss.append(0)
    lr_adjust = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, eta_min=learning_rate * 0.01, last_epoch=-1)
    for epoch in range(1, niter + 1):
        for iter_num in trange(30000 // index, desc='train, epoch:%s' % epoch):
            train_iter = train_datatset_.data_iter_index(index=index)
            
            # weights_classes = classcount(train_datatset_)
            # weights_classes = weights_classes.to(device=device, dtype=torch.float32)

            # print("Class Count of Training Set", weights_classes)
            
            for initial_image, semantic_image in train_iter:
                # print(initial_image.shape)
                # print(semantic_image.shape)
                # print(initial_image.shape)
                initial_image = initial_image.to(device=device)
                semantic_image = semantic_image.to(device=device)
                # initial_image = initial_image.cuda() #if torch.cuda.is_available() else torch.FloatTensor
                # semantic_image = semantic_image.cuda() #if torch.cuda.is_available() else torch.int64
    
                semantic_image_pred = net(initial_image)
                # print(semantic_image_pred.shape)
    
                loss = criterion(semantic_image_pred, semantic_image.long())
                # print(loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # losses = np.append(losses,loss.item())
                #mean_losses = np.append(mean_losses, np.mean(losses[max(0,iter_-100):iter_]))
                # if niter % 20 == 0:
                #     #plt.plot(mean_losses) and plt.title(label='mean_losses' ) and plt.show()
                #     plt.plot(losses) and plt.title(label='losses' ) and plt.show()
                # niter += 1
                # # if iter_ % 20 == 0:
                # #     plt.plot(mean_losses) and plt.show()
        lr_adjust.step()
    
        with torch.no_grad():
            
            weights_classes = classcount(val_datatset_)
            weights_classes = weights_classes.to(device=device, dtype=torch.float32)

            print("Class Count of Validation Set", weights_classes)
            net.eval()
            val_iter = val_datatset_.data_iter()
    
            for initial_image, semantic_image in tqdm(val_iter, desc='val'):
                # print(initial_image.shape)
                initial_image = initial_image.to(device=device)
                semantic_image = semantic_image.to(device=device)
                # initial_image = initial_image.cuda()
# #                print('mIoU: ', mIoU) #if torch.cuda.is_available() else torch.FloatTensor
#                 semantic_image = semantic_image.cuda() #if torch.cuda.is_available() else torch.int64
    
                semantic_image_pred = net(initial_image).detach()
                semantic_image_pred = F.softmax(semantic_image_pred.squeeze(), dim=0)
                semantic_image_pred = semantic_image_pred.argmax(dim=0)
    
                semantic_image = torch.squeeze(semantic_image.cpu(), 0)
                semantic_image_pred = torch.squeeze(semantic_image_pred.cpu(), 0)
    
                metric.addBatch(semantic_image_pred.float(), semantic_image)
            #     measure.addBatch(semantic_image_pred.float(), semantic_image)
                
                # pred = semantic_image_pred.cpu().numpy()
                # gt = semantic_image.cpu().numpy()
        
                # Allpred=np.append(Allpred,pred.ravel())
                # Allgt=np.append(Allgt,gt.ravel())
        
            rgb = np.asarray(255* np.transpose(initial_image.cpu().numpy()[0],(1, 2, 0)), dtype='uint8')
            #pred = np.argmax(semantic_image_pred.detach().data.cpu().numpy()[0], axis=0)
            pred= np.asarray(color_map_to_class_index.apply_GID_colormap(semantic_image_pred), dtype='uint8')
            #gt=   np.asarray((semantic_image.cpu().numpy()[0]), dtype='uint8')
            gt= np.asarray(color_map_to_class_index.apply_GID_colormap(semantic_image), dtype='uint8')
            fig = plt.figure()
            fig.add_subplot(1,3,1)
            plt.imshow(rgb)
            plt.title('RGB')
            
            fig.add_subplot(1,3,2)
            plt.imshow(pred)
            plt.title('Prediciton')
            
            fig.add_subplot(1,3,3)
            plt.imshow(gt)
            plt.title('Ground Truth')
            plt.show()
            
        # print('Unique Classes in GT: {}',np.unique(Allgt))    
        # print('Unique Classes in Pred: {}',np.unique(Allpred))
        # idx=Allgt>0
        # print(idx)
        # print(idx.shape)
        # precision, recall, fscore, support = score(Allgt[idx],Allpred[idx])
        # print('Fscore is: {}'.format(fscore))
        
        
        # cm = metric.genConfusionMatrix()
        mIoU = metric.meanIntersectionOverUnion()
        #cIoU = metric.meanIntersectionOverUnion()
        acc = metric.pixelAccuracy()
        kappa = metric.kappa()
       
        print('acc: ', acc)
        print('mIoU: ', mIoU)
        # print('cIoU: ', cIoU)
        print('kappa:', kappa)
        
        # mIoU_fg = measure.meanIntersectionOverUnion()
        # acc_fg = measure.pixelAccuracy()
        # kappa_fg = measure.kappa()
        # print('acc_fg: ', acc_fg)
        # print('mIoU_fg: ', mIoU_fg)
        # #print('mIoU_fg: ', cIoU_fg)
        # print('kappa_fg:', kappa_fg)
        
        metric.reset()
        net.train()
        
        early_stopping(1 - mIoU, net, '%s/' % out_file + 'netG.pth')
    
        if early_stopping.early_stop:
            break
    
    end = time.time()
    print('Program processed ', end - start, 's, ', (end - start)/60, 'min, ', (end - start)/3600, 'h')

    # test_datatset_ = train_dataset(test_path, time_series=band)
    # start = time.time()
    # Allpred=[]
    # Allgt=[]
    # test_iter = test_datatset_.data_iter()
    # if os.path.exists('%s/' % out_file + 'netG-a.pth'):
    #     net.load_state_dict(torch.load('%s/' % out_file + 'netG.pth'))

    # net.eval()
    # for initial_image, semantic_image in tqdm(test_iter, desc='test'):
    #     # print(initial_image.shape)
    #     initial_image = initial_image.cuda()
    #     semantic_image = semantic_image.cuda()
        
    #     # print(initial_image.shape)
    #     # print(semantic_image.shape)

    #     # semantic_image_pred = model(initial_image)
    #     semantic_image_pred = net(initial_image).detach()
    #     # print(semantic_image_pred.shape)

    #     semantic_image_pred = F.softmax(semantic_image_pred.squeeze(), dim=0)
    #     semantic_image_pred = semantic_image_pred.argmax(dim=0)

    #     semantic_image = torch.squeeze(semantic_image.cpu(), 0)
    #     semantic_image_pred = torch.squeeze(semantic_image_pred.cpu(), 0)
        
    #     # print(semantic_image.shape)
    #     # print(semantic_image_pred.shape)

    #     metric.addBatch(semantic_image_pred, semantic_image)
    #     #measure.addBatch(semantic_image_pred.float(), semantic_image)
    #     image = semantic_image_pred
        
    #     # pred = semantic_image_pred.cpu().numpy()
    #     # gt = semantic_image.cpu().numpy()
        
    #     # Allpred=np.append(Allpred,pred.ravel())
    #     # Allgt=np.append(Allgt,gt.ravel())
        
    #     rgb = np.asarray(255* np.transpose(initial_image.cpu().numpy()[0],(1, 2, 0)), dtype='uint8')
    #     #pred = np.argmax(semantic_image_pred.detach().data.cpu().numpy()[0], axis=0)
    #     pred= np.asarray(color_map_to_class_index.apply_GID_colormap(semantic_image_pred), dtype='uint8')
    #     #gt=   np.asarray((semantic_image.cpu().numpy()[0]), dtype='uint8')
    #     gt= np.asarray(color_map_to_class_index.apply_GID_colormap(semantic_image), dtype='uint8')
    #     fig = plt.figure()
    #     #plt.tick_params(left = False, labelleft = False , labelbottom = False, bottom = False)

    #     # ax = plt.gca()
    #     # ax.axes.xaxis.set_visible(False)
    #     # ax.axes.yaxis.set_visible(False)
    #     fig.add_subplot(1,3,1)
    #     plt.imshow(rgb)
    #     plt.title('RGB')
        
    #     fig.add_subplot(1,3,2)
    #     plt.imshow(pred)
    #     plt.title('Prediction')
        
    #     fig.add_subplot(1,3,3)
    #     plt.imshow(gt)
    #     plt.title('Ground Truth')
    #     plt.show()
    
    # # print('Unique Classes in GT: {}',np.unique(Allgt))    
    # # print('Unique Classes in Pred: {}',np.unique(Allpred))
    # # idx=Allgt>0
    # # print(idx)
    # # print(idx.shape)
    # # precision, recall, fscore, support = score(Allgt[idx],Allpred[idx], zero_division = 1)
    # # print('Fscore is: {}'.format(fscore))

    # end = time.time()
    # print('Program processed ', end - start, 's, ', (end - start)/60, 'min, ', (end - start)/3600, 'h')
    # # mIoU = metric.meanIntersectionOverUnion()
    # # print('mIoU: ', mIoU)
    # oa = metric.pixelAccuracy()
    # mIoU = metric.meanIntersectionOverUnion()
    # kappa = metric.kappa()
    # aa = metric.meanPixelAccuracy()
    # FWIoU = metric.Frequency_Weighted_Intersection_over_Union()
    # PixelAccuracy = metric.classPixelAccuracy()
    # RecallAccuracy = metric.classRecallAccuracy()
    # F1Score = metric.F1Score()
    # print('oa: ', oa)
    # print('mIoU: ', mIoU)
    # print('kappa', kappa)
    # print('aa: ', aa)
    # print('FWIoU: ', FWIoU)
    # print('PixelAccuracy:', PixelAccuracy)
    # print('RecallAccuracy:', RecallAccuracy)
    # print('F1Score: ', F1Score)