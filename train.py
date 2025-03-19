import torch.optim as optim
import torch.nn.functional as F
import random
import clip
import torch
import math
import time
import torch.nn as nn
from con_losses import SupConLoss


def train(args, dataloader, DEVICE, model, epoch, label_name, text_queue):
    if args.algorithm == 'CSMamba':
        CNN_correct = 0
        LEARNING_RATE = args.lr / math.pow((1 + 10 * (epoch - 1) / epoch), 0.75)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        text_templates = [
            "A hyperspectral image of {}",
            "This hyperspectral capture shows {}",
            "An image revealing the spectral properties of {}",
            "Hyperspectral imaging of {}",
            # "Spectral data collected from {}",
            "An analysis image of {} using hyperspectral technology",
            # "A detailed spectral view of {}"
        ]
        iter_source = iter(dataloader)
        num_iter = len(dataloader)
        t1 = time.time()
        for i in range(1, num_iter):
            model.train()
            data_src, label_src = next(iter_source)
            data_src, label_src = data_src.to(DEVICE), label_src.to(DEVICE)
            label_src = label_src - 1
            text = []
            for k in label_src:
                label = label_name[k]
                if random.choice([True, False]):
                    template = random.choice(text_templates)
                    text.append(clip.tokenize(template.format(label)))

                else:
                    text.append(clip.tokenize(text_queue[label][0]))

            text_src = torch.cat([j for j in text]).to(DEVICE)
            optimizer.zero_grad()

            # loss_tcl, label_src_pred, reg = model(data_src, text_src, label_src)
            loss_tcl, label_src_pred = model(data_src, text_src, label_src)

            loss_cls = F.nll_loss(F.log_softmax(label_src_pred, dim=1), label_src.long())
            loss = loss_cls + args.alpha * loss_tcl
            # loss = loss_cls + args.alpha * loss_tcl + 0.1 * F.cross_entropy(reg, data_src[:, :, 8, 8])
            loss.backward()
            optimizer.step()

            pred = label_src_pred.data.max(1)[1]
            CNN_correct += pred.eq(label_src.data.view_as(pred)).cpu().sum()
            if i % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]'.format(epoch, i * len(data_src), len(dataloader.dataset),
                                                                 100. * i / num_iter))
                print(
                    'loss:{:.6f},loss_tcl:{:.6f},loss_cls:{:.6f}'.format(loss.item(), loss_tcl.item(), loss_cls.item()))
        t2 = time.time()
        CCN_acc = CNN_correct.item() / len(dataloader.dataset)
        print('[epoch: {:4}]  Train Accuracy: {:.4f} | train sample number: {:6} Trainingtime:{:.4f}'.format(epoch,
                                                                                                             CCN_acc,
                                                                                                             len(dataloader.dataset),
                                                                                                             t2 - t1))
        return model

def train_GCN(args, dataloader, DEVICE, model, epoch, index_train):
    if args.algorithm == 'GCN':
        CNN_correct = 0
        LEARNING_RATE = args.lr / math.pow((1 + 10 * (epoch - 1) / epoch), 0.75)
        criterion = nn.CrossEntropyLoss().cuda()
        con_criterion = SupConLoss(device=args.cuda)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=args.weight_decay)
        iter_source = iter(dataloader)
        num_iter = len(dataloader)
        t1 = time.time()
        for i in range(1, num_iter):
            model.train()
            A, batch_data, batch_target = next(iter_source)
            # print(batch_data.shape)
            A, batch_data, batch_target = A.to(DEVICE), batch_data.to(DEVICE), batch_target.to(DEVICE)-1
            optimizer.zero_grad()
            batch_pred, cnn_pred, sage_pred, gcn_proj, cl = model(batch_data, A, batch_target, index_train)
            loss_con = con_criterion(gcn_proj, batch_target, adv=False)
            loss_cls = criterion(batch_pred, batch_target) + args.alpha * criterion(sage_pred, batch_target) + args.alpha * criterion(cnn_pred, batch_target)
            loss = loss_cls + args.alpha * loss_con + args.alpha * cl

            loss.backward()
            optimizer.step()
            pred = batch_pred.data.max(1)[1]
            CNN_correct += pred.eq(batch_target.data.view_as(pred)).cpu().sum()
            if i % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)],loss:{:.6f}'.format(epoch, i * len(batch_data), len(dataloader.dataset),
                                                                 100. * i / num_iter, loss.item()))
                # print(
                #     'loss:{:.6f}'.format(loss.item()))
        t2 = time.time()
        CCN_acc = CNN_correct.item() / len(dataloader.dataset)
        print('[epoch: {:4}]  Train Accuracy: {:.4f} | train sample number: {:6} Trainingtime:{:.4f}'.format(epoch,
                                                                                                             CCN_acc,
                                                                                                             len(dataloader.dataset),
                                                                                                             t2 - t1))
        return model

