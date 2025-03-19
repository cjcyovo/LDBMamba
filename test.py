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
        # reg_list = []
        # im_mat2 = loadmat('datasets/Houston/Houston18_7gt.mat')
        # im_mat2 = loadmat('datasets/Pavia/paviaC_7gt.mat')
        # im_mat2 = loadmat('datasets/Indiana/IndianaTD_7gt.mat')
        # image2 = im_mat2["map"]
        # non_zero_indices = np.argwhere(image2 != 0)
        t_test_begin = time.time()
        with (torch.no_grad()):
            for data, label in dataloader:
                data, label = data.to(DEVICE), label.to(DEVICE)
                label = label - 1
                label_src_pred = model(data, label, label)
                # reg = label_src_pred
                pred = label_src_pred.data.max(1)[1]
                pred_list.append(pred.cpu().numpy())
                label_list.append(label.cpu().numpy())
                loss += F.nll_loss(F.log_softmax(label_src_pred, dim=1), label.long()).item()
                correct += pred.eq(label.data.view_as(pred)).cpu().sum()
        #
        #         reg = reg.to("cpu")
        #
        #         reg = reg.numpy()
        #         reg_list.append(reg)
        # regs = np.zeros((210, 954, 7))
        # regs = np.zeros((1096, 715, 7))
        # regs = np.zeros((300, 400, 7))
        # reg_list = [item for sublist in reg_list for item in sublist]
        # for idx, reg in enumerate(reg_list):
        #     # 获取当前 reg 对应的非零位置
        #     x, y = non_zero_indices[idx]
        #
        #     # 将 reg 放入 regs 中的对应位置
        #     regs[x, y, :] = reg
        t_test_finish = time.time()
        loss /= len(dataloader)
        print(
            'Average test loss: {:.4f},test Accuracy: {}/{} ({:.2f}%), | test sample number: {:6},   test time: {:4f}s\n'.format(
                loss, correct, len(dataloader.dataset), 100. * correct / len(dataloader.dataset),
                len(dataloader.dataset), t_test_finish - t_test_begin))
        return correct, pred_list, label_list
        # return correct, correct.item() / len(dataloader.dataset), pred_list, label_list

def test_GCN(args, dataloader, DEVICE, model, epoch, index_test):
    if args.algorithm == 'GCN':
        model.eval()
        loss = 0
        correct = 0
        pred_list, label_list = [], []
        with (torch.no_grad()):
            for A, data, label in dataloader:
                A, data, label, = A.to(DEVICE), data.to(DEVICE), label.to(DEVICE)-1,
                label_src_pred = model(data, A, label, index_test)
                # reg = label_src_pred
                pred = label_src_pred.data.max(1)[1]
                pred_list.append(pred.cpu().numpy())
                label_list.append(label.cpu().numpy())
                loss += F.nll_loss(F.log_softmax(label_src_pred, dim=1), label.long()).item()
                # print(loss)
                correct += pred.eq(label.data.view_as(pred)).cpu().sum()
        loss /= len(dataloader)
        print(
            'Average test loss: {:.4f},test Accuracy: {}/{} ({:.2f}%), | test sample number: {:4f}\n'.format(
                loss, correct, len(dataloader.dataset), 100. * correct / len(dataloader.dataset),
                len(dataloader.dataset)))
        return correct, pred_list, label_list