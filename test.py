import torch
import torch.nn.functional as F
import numpy as np
from scipy.io import loadmat
import time


def test(args, dataloader, DEVICE, model, epoch):
    if args.algorithm == 'CSMamba':
        model.eval()
        loss = 0
        correct = 0
        pred_list, label_list = [], []
        t_test_begin = time.time()
        with (torch.no_grad()):
            for data, label in dataloader:
                data, label = data.to(DEVICE), label.to(DEVICE)
                label = label - 1
                label_src_pred = model(data, label, label)
                pred = label_src_pred.data.max(1)[1]
                pred_list.append(pred.cpu().numpy())
                label_list.append(label.cpu().numpy())
                loss += F.nll_loss(F.log_softmax(label_src_pred, dim=1), label.long()).item()
                correct += pred.eq(label.data.view_as(pred)).cpu().sum()
        t_test_finish = time.time()
        loss /= len(dataloader)
        print(
            'Average test loss: {:.4f},test Accuracy: {}/{} ({:.2f}%), | test sample number: {:6},   test time: {:4f}s\n'.format(
                loss, correct, len(dataloader.dataset), 100. * correct / len(dataloader.dataset),
                len(dataloader.dataset), t_test_finish - t_test_begin))
        return correct, pred_list, label_list
