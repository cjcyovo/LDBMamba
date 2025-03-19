import torch.optim as optim
import torch.nn.functional as F
import random
import clip
import torch
import math
import time
import torch.nn as nn


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
            "An analysis image of {} using hyperspectral technology",
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
            loss_tcl, label_src_pred = model(data_src, text_src, label_src)
            loss_cls = F.nll_loss(F.log_softmax(label_src_pred, dim=1), label_src.long())
            loss = loss_cls + args.alpha * loss_tcl
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
