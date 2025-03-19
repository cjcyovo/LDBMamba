import CSMamba
def get_model(args):
    if args.algorithm == 'CSMamba':
        return CSMamba(n_bands=args.n_bands, patch_size=args.patch_size, spa_size=args.spa_size, spe_size=args.spe_size, layer_d_model=args.layer_d_model)
