import torch
import numpy as np
import imageio
from scipy import io
import os
import spectral
import random
import sklearn.model_selection
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.patches as mpatches
from PIL import Image
from scipy.io import loadmat

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def build_ground_truth_image(dataset, pred_list, savename):
    if dataset == 'Houston':
        labels = io.loadmat('datasets/Houston/Houston18_7gt.mat')['map']
        gt_list = ['undefined', "grass healthy", "grass stressed", "trees",
                   "water", "residential buildings",
                   "non-residential buildings", "road"]

    elif dataset == 'Pavia':
        labels = io.loadmat('datasets/Pavia/paviaC_7gt.mat')['map']
        gt_list = ['undefined', "tree", "asphalt", "brick",
                   "bitumen", "shadow", 'meadow', 'bare soil']

    elif dataset == 'Indiana':
        labels = io.loadmat('datasets/Indiana/IndianaTD_7gt.mat')['map']
        gt_list = ['undefined', "Concrete/Asphalt", "Corn-CleanTill", "Corn-CleanTill-EW",
                   "Orchard", "Soybeans-CleanTill", "Soybeans-CleanTill-EW", "Wheat"]

    else:
        gt_list = ['undefined']
    height = labels.shape[0]
    width = labels.shape[1]
    outputs = np.zeros((height, width))
    a = 0
    b = 0
    for i in range(height):
        for j in range(width):
            if labels[i, j] == 0:
                continue
            else:
                outputs[i, j] = pred_list[a][b] + 1
                b = b + 1
                if b == 256:
                    b = 0
                    a = a + 1
    gt = outputs.astype(int)

    color_list = [(0, 0, 0), (255, 0, 0), (255, 255, 0), (155, 48, 255), (0, 0, 205),
                  (255, 165, 0), (0, 255, 0), (0, 139, 139), (255, 240, 245),
                  (139, 0, 0), (230, 230, 250), (155, 205, 155), (25, 25, 112),
                  (255, 193, 37), (255, 106, 106), (205, 104, 57), (160, 32, 240)]
    color_hex_list = ["#FFFFFF", "#FF0000", "#FFFF00", "#9B30FF", "#0000CD",
                      "#FFA500", "#00FF00", "#008B8B", "#FFF0F5",
                      "#8B0000", "#E6E6FA", "#9BCD9B", "#191970",
                      "#FFC125", "#FF6A6A", "#CD6839", "#A020F0", ]

    gt_rgb = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)
    for x in range(gt.shape[0]):
        for y in range(gt.shape[1]):
            gt_rgb[x, y, :] = color_list[gt[x, y]]

    gt_classes = set(gt.flatten().tolist())
    patch_list = []
    for i in range(1, len(gt_classes)):
        patch_list.append(mpatches.Patch(color=color_hex_list[i], label=gt_list[i]))
    gt_rgb = Image.fromarray(gt_rgb)
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    plt.imshow(gt_rgb)
    plt.savefig(savename + '.svg', format='svg', dpi=300)


def seed_worker(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    random.seed(seed)
    np.random.seed(seed)  # Numpy module.
    #random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def check_dataset(args):

    if args.dataset == 'Pavia':
        args.data_path = './datasets/Pavia/'
        args.source_name = 'paviaU'
        args.target_name = 'paviaC'
        args.re_ratio = 1
    if args.dataset == 'Indiana':
        args.data_path = './datasets/Indiana/'
        args.source_name = 'IndianaSD'
        args.target_name = 'IndianaTD'
        args.re_ratio = 1
    if args.dataset == 'HyRANK':
        args.data_path = './datasets/HyRANK/'
        args.source_name = 'Dioni'
        args.target_name = 'Loukia'
        args.re_ratio = 1
    return args

def open_file(dataset):
    _, ext = os.path.splitext(dataset)
    ext = ext.lower()
    if ext == '.mat':
        # Load Matlab array
        return io.loadmat(dataset)
        # return h5py.File(dataset,'r')
    elif ext == '.tif' or ext == '.tiff':
        # Load TIFF file
        return imageio.imread(dataset)
    elif ext == '.hdr':
        img = spectral.open_image(dataset)
        return img.load()
    else:
        raise ValueError("Unknown file format: {}".format(ext))


def get_device(ordinal):
    # Use GPU ?
    if ordinal < 0:
        print("Computation on CPU")
        device = torch.device('cpu')
    elif torch.cuda.is_available():
        print("Computation on CUDA GPU device {}".format(ordinal))
        device = torch.device('cuda:{}'.format(ordinal))
    else:
        print("/!\\ CUDA was requested but is not available! Computation will go on CPU. /!\\")
        device = torch.device('cpu')
    return device


def sample_gt(gt, train_size, mode='random'):
    """Extract a fixed percentage of samples from an array of labels.

    Args:
        gt: a 2D array of int labels
        percentage: [0, 1] float
    Returns:
        train_gt, test_gt: 2D arrays of int labels

    """
    indices = np.nonzero(gt)
    X = list(zip(*indices)) # x,y features
    y = gt[indices].ravel() # classes
    train_gt = np.zeros_like(gt)
    test_gt = np.zeros_like(gt)
    if train_size > 1:
       train_size = int(train_size)
    train_label = []
    test_label = []
    if mode == 'random':
        if train_size == 1:
            random.shuffle(X)
            train_indices = [list(t) for t in zip(*X)]
            [train_label.append(i) for i in gt[tuple(train_indices)]]
            train_set = np.column_stack((train_indices[0],train_indices[1],train_label))
            train_gt[tuple(train_indices)] = gt[tuple(train_indices)]
            test_gt = []
            test_set = []
        else:
            train_indices, test_indices = sklearn.model_selection.train_test_split(X, train_size=train_size, stratify=y, random_state=23)
            train_indices = [list(t) for t in zip(*train_indices)]
            test_indices = [list(t) for t in zip(*test_indices)]
            train_gt[tuple(train_indices)] = gt[tuple(train_indices)]
            test_gt[tuple(test_indices)] = gt[tuple(test_indices)]

            [train_label.append(i) for i in gt[tuple(train_indices)]]
            train_set = np.column_stack((train_indices[0],train_indices[1],train_label))
            [test_label.append(i) for i in gt[tuple(test_indices)]]
            test_set = np.column_stack((test_indices[0],test_indices[1],test_label))

    elif mode == 'disjoint':
        train_gt = np.copy(gt)
        test_gt = np.copy(gt)
        for c in np.unique(gt):
            mask = gt == c
            for x in range(gt.shape[0]):
                first_half_count = np.count_nonzero(mask[:x, :])
                second_half_count = np.count_nonzero(mask[x:, :])
                try:
                    ratio = first_half_count / second_half_count
                    if ratio > 0.9 * train_size and ratio < 1.1 * train_size:
                        break
                except ZeroDivisionError:
                    continue
            mask[:x, :] = 0
            train_gt[mask] = 0

        test_gt[train_gt > 0] = 0
    else:
        raise ValueError("{} sampling is not implemented yet.".format(mode))
    return train_gt, test_gt, train_set, test_set


def metrics(prediction, target, ignored_labels=[], n_classes=None):
    """Compute and print metrics (accuracy, confusion matrix and F1 scores).

    Args:
        prediction: list of predicted labels
        target: list of target labels
        ignored_labels (optional): list of labels to ignore, e.g. 0 for undef
        n_classes (optional): number of classes, max(target) by default
    Returns:
        accuracy, F1 score by class, confusion matrix
    """
    ignored_mask = np.zeros(target.shape[:2], dtype=np.bool_)
    for l in ignored_labels:
        ignored_mask[target == l] = True
    ignored_mask = ~ignored_mask
    #target = target[ignored_mask] -1
    # target = target[ignored_mask]
    # prediction = prediction[ignored_mask]

    results = {}

    n_classes = np.max(target) + 1 if n_classes is None else n_classes

    cm = confusion_matrix(
        target,
        prediction,
        labels=range(n_classes))

    results["Confusion_matrix"] = cm

    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    results["TPR"] = TPR
    # Compute global accuracy
    total = np.sum(cm)
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)

    results["Accuracy"] = accuracy

    # Compute F1 score
    F1scores = np.zeros(len(cm))
    for i in range(len(cm)):
        try:
            F1 = 2 * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
        except ZeroDivisionError:
            F1 = 0.
        F1scores[i] = F1

    results["F1_scores"] = F1scores

    # Compute kappa coefficient
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / \
        float(total * total)
    kappa = (pa - pe) / (1 - pe)
    results["Kappa"] = kappa

    results["prediction"] = prediction
    results["label"] = target

    return results
