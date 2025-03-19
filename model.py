from algs.CSMamba import CSMamba
from algs.DGCN import GCN
def get_model(args):
    if args.algorithm == 'CSMamba':
        return CSMamba(n_bands=args.n_bands, patch_size=args.patch_size, spa_size=args.spa_size, spe_size=args.spe_size, layer_d_model=args.layer_d_model)

    elif args.algorithm == 'GCN':
        return GCN(height=args.patch_size, width=args.patch_size, channel=args.n_bands, class_count=args.n_classes)