import configargparse
import torch
from utils import seed_worker, check_dataset, get_device, metrics, build_ground_truth_image
from datasets import create_dataloader
import numpy as np
from train import train
from test import test
from model import LDBMamba
from sklearn.metrics import classification_report
import os
from model import get_model
import scipy

def get_parser():
    """Get default arguments."""
    parser = configargparse.ArgumentParser(
        description="Domain Generalization config parser",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )
    # general configuration
    parser.add("--config", is_config_file=True, help="config file path")
    parser.add("--seed", type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--algorithm', type=str)
    parser.add_argument('--cuda', type=int, default=0,
                        help="Specify CUDA device (defaults to -1, which learns on CPU)")
    # data loading related
    parser.add_argument('--dataset', type=str, default='Houston', help='the dataset to select')
    parser.add_argument('--data_path', type=str, default="./datasets/Houston/")
    parser.add_argument('--source_name', type=str, default="Houston13")
    parser.add_argument('--target_name', type=str, default="Houston18")
    parser.add_argument('--patch_size', type=int, default=15)
    parser.add_argument('--training_sample_ratio', type=float, default=0.8,
                        help='training sample ratio')
    parser.add_argument('--re_ratio', type=int, default=5,
                        help='multiple of of data augmentation')
    parser.add_argument('--flip_augmentation', action='store_true', default=False,
                        help="Random flips (if patch_size > 1)")
    parser.add_argument('--radiation_augmentation', action='store_true', default=False,
                        help="Random radiation noise (illumination)")
    parser.add_argument('--mixture_augmentation', action='store_true', default=False,
                        help="Random mixes between spectral")
    # training related
    parser.add_argument('--batch_size', type=int, default=256,
                        help="Batch size (optional, if absent will be set by the model")
    parser.add_argument('--lr', type=float, default=1e-2,
                             help="Learning rate, set by the model if not specified.")
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='the Adam weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--class_balancing', action='store_true',
                             help="Inverse median frequency class balancing (default = False)")
    parser.add_argument('--test_stride', type=int, default=1,
                             help="Sliding window step stride during inference (default = 1)")
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--l2_decay', type=float, default=1e-4,
                        help='the L2  weight decay')
    parser.add_argument('--num_epoch', type=int, default=400,
                        help='the number of epoch')
    parser.add_argument('--num_trials', type=int, default=1,
                        help='the number of epoch')
    parser.add_argument('--alpha', type=float, default=0.01,
                        help='loss_tcl')
    #model related
    parser.add_argument('--spa_size', type=int, default=3,
                        help="spa patchfy size")
    parser.add_argument('--spe_size', type=int, default=4,
                        help="spe patchfy size")
    parser.add_argument('--layer_d_model', nargs='+', type=int, default=[64, 64, 32, 16, 8])
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    DEVICE = get_device(args.cuda)
    seed_worker(args.seed)
    check_dataset(args)
    if args.algorithm == 'CSMamba':
        train_loader, test_loader, IGNORED_LABELS, LABEL_VALUES_tar, label_true_queue = create_dataloader(
            args, DEVICE)
        print(len(train_loader), len(train_loader.dataset), len(test_loader), len(test_loader.dataset))
        net = get_model(args).to(DEVICE)
        print(args)
        total_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f'{total_trainable_params / (1024 * 1024):.2f}M training parameters.')
        correct = 0
        for epoch in range(1, args.num_epoch + 1):
            net = train(args, train_loader, DEVICE, net, epoch, LABEL_VALUES_tar, label_true_queue)

            t_correct, pred, label = test(args, test_loader, DEVICE, net, epoch)

            if t_correct > correct:
                correct = t_correct
                results = metrics(np.concatenate(pred), np.concatenate(label), ignored_labels=IGNORED_LABELS,
                                  n_classes=7)
                kappa = results['Kappa']
                print(classification_report(np.concatenate(pred), np.concatenate(label), target_names=LABEL_VALUES_tar,
                                            digits=6))

            print('source: {} to target: {} max correct: {} max accuracy{: .2f}% kappa{: .2f}\n'.format(
                args.source_name, args.target_name, correct, 100. * correct / len(test_loader.dataset), 100. * kappa))
            

if __name__ == "__main__":
    main()
