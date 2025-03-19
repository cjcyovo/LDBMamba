import numpy as np
import os
from utils import open_file, sample_gt, seed_worker, augment_data, GET_A2, gain_neighborhood_pixel, get_index_edges
from tqdm import tqdm
import torch
import torch.utils.data as data

try:
    # Python 3
    from urllib.request import urlretrieve
except ImportError:
    # Python 2
    from urllib import urlretrieve
DATASETS_CONFIG = {
    'Houston13': {
        'img': 'Houston13.mat',
        'gt': 'Houston13_7gt.mat',
    },
    'Houston18': {
        'img': 'Houston18.mat',
        'gt': 'Houston18_7gt.mat',
    },
    'paviaU': {
        'img': 'paviaU.mat',
        'gt': 'paviaU_7gt.mat',
    },
    'paviaC': {
        'img': 'paviaC.mat',
        'gt': 'paviaC_7gt.mat',
    },
    'IndianaSD': {
        'img': 'IndianaSD.mat',
        'gt': 'IndianaSD_7gt.mat',
    },
    'IndianaTD': {
        'img': 'IndianaTD.mat',
        'gt': 'IndianaTD_7gt.mat',
    },
    'Dioni':{
        'img': 'Dioni.mat',
        'gt': 'Dioni_gt_out68.mat',
    },
    'Loukia': {
        'img': 'Loukia.mat',
        'gt': 'Loukia_gt_out68.mat',
    },
}

try:
    from custom_datasets import CUSTOM_DATASETS_CONFIG

    DATASETS_CONFIG.update(CUSTOM_DATASETS_CONFIG)
except ImportError:
    pass

def get_dataset(dataset_name, target_folder="./", datasets=DATASETS_CONFIG):
    """ Gets the dataset specified by name and return the related components.
    Args:
        dataset_name: string with the name of the dataset
        target_folder (optional): folder to store the datasets, defaults to ./
        datasets (optional): dataset configuration dictionary, defaults to prebuilt one
    Returns:
        img: 3D hyperspectral image (WxHxB)
        gt: 2D int array of labels
        label_values: list of class names
        ignored_labels: list of int classes to ignore
        rgb_bands: int tuple that correspond to red, green and blue bands
    """
    palette = None

    if dataset_name not in datasets.keys():
        raise ValueError("{} dataset is unknown.".format(dataset_name))

    dataset = datasets[dataset_name]

    folder = target_folder  # + datasets[dataset_name].get('folder', dataset_name + '/')
    if dataset.get('download', False):
        # Download the dataset if is not present
        if not os.path.isdir(folder):
            os.mkdir(folder)
        for url in datasets[dataset_name]['urls']:
            # download the files
            filename = url.split('/')[-1]
            if not os.path.exists(folder + filename):
                with TqdmUpTo(unit='B', unit_scale=True, miniters=1,
                              desc="Downloading {}".format(filename)) as t:
                    urlretrieve(url, filename=folder + filename,
                                reporthook=t.update_to)
    elif not os.path.isdir(folder):
        print("WARNING: {} is not downloadable.".format(dataset_name))

    if dataset_name == 'Houston13':
        # Load the image
        img = open_file(folder + 'Houston13.mat')['ori_data']  # (210, 954, 48)

        rgb_bands = [13, 20, 33]

        gt = open_file(folder + 'Houston13_7gt.mat')['map']  # (210, 954)

        label_values = ["grass healthy", "grass stressed", "trees",
                        "water", "residential buildings",
                        "non-residential buildings", "road"]

        label_true_queue = {"grass healthy": ['The grass healthy is lush'],
                       "grass stressed": ['The grass stressed by the road appears pale'],
                       "trees": ['The trees grow steadily along the road'],
                       "water": ['Water appears smooth with a dark blue or black color'],
                       "residential buildings": ['Residential buildings arranged neatly'],
                       "non-residential buildings": ['Non-residential buildings vary in shape'],
                       "road": ['Roads divide buildings into blocks']}


        label_false_queue = {"grass healthy": ['The grass healthy is lush'],
                       "grass stressed": ['The grass stressed by the road appears pale'],
                       "trees": ['The trees grow steadily along the road'],
                       "water": ['Water appears smooth with a dark blue or black color'],
                       "residential buildings": ['Residential buildings arranged neatly'],
                       "non-residential buildings": ['Non-residential buildings vary in shape'],
                       "road": ['Roads divide buildings into blocks']}


        ignored_labels = [0]

    elif dataset_name == 'Houston18':
        # Load the image
        img = open_file(folder + 'Houston18.mat')['ori_data']  # (210, 954, 48)

        rgb_bands = [13, 20, 33]

        gt = open_file(folder + 'Houston18_7gt.mat')['map']  # (210, 954)

        label_values = ["grass healthy", "grass stressed", "trees",
                        "water", "residential buildings",
                        "non-residential buildings", "road"]
        label_true_queue = {"grass healthy": ['The grass healthy is lush'],
                       "grass stressed": ['The grass stressed by the road appears pale'],
                       "trees": ['The trees grow steadily along the road'],
                       "water": ['Water appears smooth with a dark blue or black color'],
                       "residential buildings": ['Residential buildings arranged neatly'],
                       "non-residential buildings": ['Non-residential buildings vary in shape'],
                       "road": ['Roads divide buildings into blocks']}


        label_false_queue = {"grass healthy": ['The grass healthy is lush'],
                       "grass stressed": ['The grass stressed by the road appears pale'],
                       "trees": ['The trees grow steadily along the road'],
                       "water": ['Water appears smooth with a dark blue or black color'],
                       "residential buildings": ['Residential buildings arranged neatly'],
                       "non-residential buildings": ['Non-residential buildings vary in shape'],
                       "road": ['Roads divide buildings into blocks']}
        ignored_labels = [0]

    elif dataset_name == 'paviaU':
        # Load the image
        img = open_file(folder + 'paviaU.mat')['ori_data']  # (610, 340, 102)

        rgb_bands = [20, 30, 30]

        gt = open_file(folder + 'paviaU_7gt.mat')['map']  # (610, 340)

        label_values = ["tree", "asphalt", "brick",
                        "bitumen", "shadow", 'meadow', 'bare soil']
        label_true_queue = {"tree": ['The trees grow steadily along the road'],
                       "asphalt": ['Asphalt is used to pave roads'],
                       "brick": ['A brick is a type of construction material'],
                       "bitumen": ['Bitumen is a material for building surfaces'],
                       "shadow": ['Shadows will appear on the backlight of the building'],
                       "meadow": ['Meadow is a land covered with grass'],
                       "bare soil": ['No vegetation on the surface of bare soil']}
        label_false_queue = {"tree": ['The trees grow steadily along the road'],
                       "asphalt": ['Asphalt is used to pave roads'],
                       "brick": ['A brick is a type of construction material'],
                       "bitumen": ['Bitumen is a material for building surfaces'],
                       "shadow": ['Shadows will appear on the backlight of the building'],
                       "meadow": ['Meadow is a land covered with grass'],
                       "bare soil": ['No vegetation on the surface of bare soil']}

        ignored_labels = [0]
    elif dataset_name == 'paviaC':
        # Load the image
        img = open_file(folder + 'paviaC.mat')['ori_data']  # (1096, 715, 102)

        rgb_bands = [20, 30, 30]

        gt = open_file(folder + 'paviaC_7gt.mat')['map']  # (1096, 715)

        label_values = ["tree", "asphalt", "brick",
                        "bitumen", "shadow", 'meadow', 'bare soil']
        label_true_queue = {"tree": ['The trees grow steadily along the road'],
                            "asphalt": ['Asphalt is used to pave roads'],
                            "brick": ['A brick is a type of construction material'],
                            "bitumen": ['Bitumen is a material for building surfaces'],
                            "shadow": ['Shadows will appear on the backlight of the building'],
                            "meadow": ['Meadow is a land covered with grass'],
                            "bare soil": ['No vegetation on the surface of bare soil']}
        label_false_queue = {"tree": ['The trees grow steadily along the road'],
                             "asphalt": ['Asphalt is used to pave roads'],
                             "brick": ['A brick is a type of construction material'],
                             "bitumen": ['Bitumen is a material for building surfaces'],
                             "shadow": ['Shadows will appear on the backlight of the building'],
                             "meadow": ['Meadow is a land covered with grass'],
                             "bare soil": ['No vegetation on the surface of bare soil']}

        ignored_labels = [0]
    elif dataset_name == 'IndianaSD':
        # Load the image
        img = open_file(folder + 'IndianaSD.mat')['ori_data']  # (400, 300, 220)


        rgb_bands = [9, 19, 29]

        gt = open_file(folder + 'IndianaSD_7gt.mat')['map']  # (400, 300)
        print(np.count_nonzero(gt))
        label_values = ["Concrete/Asphalt", "Corn-CleanTill", "Corn-CleanTill-EW",
                        "Orchard", "Soybeans-CleanTill", "Soybeans-CleanTill-EW", "Wheat"]
        label_true_queue = {"Concrete/Asphalt": ['No crops on the surfaces of Concrete or Asphalt'],
                       "Corn-CleanTill": ['Corn-CleanTill planted with corn'],
                       "Corn-CleanTill-EW": ['Corn-CleanTill-EW planted with early maturing maize'],
                       "Orchard": ['The orchard is full of fruit trees'],
                       "Soybeans-CleanTill": ['Soybeans-CleanTill planted with Soybeans'],
                       "Soybeans-CleanTill-EW": ['Soybeans-CleanTill-EW planted with early maturing Soybeans'],
                       "Wheat": ['Wheat is an important food crop']}
        label_false_queue = {"Concrete/Asphalt": ['No crops on the surfaces of Concrete or Asphalt'],
                       "Corn-CleanTill": ['Corn-CleanTill planted with corn'],
                       "Corn-CleanTill-EW": ['Corn-CleanTill-EW planted with early maturing maize'],
                       "Orchard": ['The orchard is full of fruit trees'],
                       "Soybeans-CleanTill": ['Soybeans-CleanTill planted with Soybeans'],
                       "Soybeans-CleanTill-EW": ['Soybeans-CleanTill-EW planted with early maturing Soybeans'],
                       "Wheat": ['Wheat is an important food crop']}
        ignored_labels = [0]
    elif dataset_name == 'IndianaTD':
        # Load the image
        img = open_file(folder + 'IndianaTD.mat')['ori_data']  # (400, 300, 220)

        rgb_bands = [9, 19, 29]

        gt = open_file(folder + 'IndianaTD_7gt.mat')['map']  # (400, 300)
        print(np.count_nonzero(gt))
        label_values = ["Concrete/Asphalt", "Corn-CleanTill", "Corn-CleanTill-EW",
                        "Orchard", "Soybeans-CleanTill", "Soybeans-CleanTill-EW", "Wheat"]
        label_true_queue = {"Concrete/Asphalt": ['No crops on the surfaces of Concrete or Asphalt'],
                            "Corn-CleanTill": ['Corn-CleanTill planted with corn'],
                            "Corn-CleanTill-EW": ['Corn-CleanTill-EW planted with early maturing maize'],
                            "Orchard": ['The orchard is full of fruit trees'],
                            "Soybeans-CleanTill": ['Soybeans-CleanTill planted with Soybeans'],
                            "Soybeans-CleanTill-EW": ['Soybeans-CleanTill-EW planted with early maturing Soybeans'],
                            "Wheat": ['Wheat is an important food crop']}
        label_false_queue = {"Concrete/Asphalt": ['No crops on the surfaces of Concrete or Asphalt'],
                             "Corn-CleanTill": ['Corn-CleanTill planted with corn'],
                             "Corn-CleanTill-EW": ['Corn-CleanTill-EW planted with early maturing maize'],
                             "Orchard": ['The orchard is full of fruit trees'],
                             "Soybeans-CleanTill": ['Soybeans-CleanTill planted with Soybeans'],
                             "Soybeans-CleanTill-EW": ['Soybeans-CleanTill-EW planted with early maturing Soybeans'],
                             "Wheat": ['Wheat is an important food crop']}
        ignored_labels = [0]
    elif dataset_name =='Dioni':
        img = open_file(folder + 'Dioni.mat')['ori_data']

        rgb_bands = (43, 21, 11)

        gt = open_file(folder + 'Dioni_gt_out68.mat')['map']
        label_values = ["Dense Urban Fabric", "Mineral Extraction Sites", "Non Irrigated Arable Land", "Fruit Trees",
                        "Olive Groves", "Coniferous Forest", "Dense Sderophyllous Vegetation",
                        "Sparce Sderophyllous Vegetation", "Sparcely Vegetated Areas", "Rocks and Sand", "Water", "Coastal Water"]
        ignored_labels = [0]
        label_true_queue = {}
        label_false_queue= {}
    elif dataset_name == 'Loukia':
        img = open_file(folder + 'Loukia.mat')['ori_data'] #(249, 945, 176)

        rgb_bands = (43, 21, 11)

        gt = open_file(folder + 'Loukia_gt_out68.mat')['map']
        label_values = ["Dense Urban Fabric", "Mineral Extraction Sites", "Non Irrigated Arable Land", "Fruit Trees",
                        "Olive Groves", "Coniferous Forest", "Dense Sderophyllous Vegetation",
                        "Sparce Sderophyllous Vegetation", "Sparcely Vegetated Areas", "Rocks and Sand", "Water",
                        "Coastal Water"]
        ignored_labels = [0]
        label_true_queue = {}
        label_false_queue = {}
    else:
        # Custom dataset
        img, gt, rgb_bands, ignored_labels, label_values, palette = CUSTOM_DATASETS_CONFIG[dataset_name]['loader'](
            folder)

    # Filter NaN out
    nan_mask = np.isnan(img.sum(axis=-1))
    if np.count_nonzero(nan_mask) > 0:
        print(
            "Warning: NaN have been found in the data. It is preferable to remove them beforehand. Learning on NaN data is disabled.")
    img[nan_mask] = 0
    gt[nan_mask] = 0
    ignored_labels.append(0)

    ignored_labels = list(set(ignored_labels))  # 确保ignored_labels不含重复的标签
    # Normalization
    img = np.asarray(img, dtype='float32')

    m, n, d = img.shape[0], img.shape[1], img.shape[2]
    img = img.reshape((m * n, -1))
    img = img / img.max()
    img_temp = np.sqrt(np.asarray((img ** 2).sum(1)))
    img_temp = np.expand_dims(img_temp, axis=1)
    img_temp = img_temp.repeat(d, axis=1)
    img_temp[img_temp == 0] = 1
    img = img / img_temp
    img = np.reshape(img, (m, n, -1))

    return img, gt, label_values, label_true_queue, label_false_queue, ignored_labels, rgb_bands, palette

class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""

    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize

class HyperX(torch.utils.data.Dataset):
    """ Generic class for a hyperspectral scene """

    def __init__(self, data, gt, transform=None, **hyperparams):
        """
        Args:
            data: 3D hyperspectral image
            gt: 2D array of labels
            patch_size: int, size of the spatial neighbourhood
            center_pixel: bool, set to True to consider only the label of the
                          center pixel
            data_augmentation: bool, set to True to perform random flips
            supervision: 'full' or 'semi' supervised algorithms
        """
        super(HyperX, self).__init__()
        self.transform = transform
        self.data = data
        self.label = gt
        self.patch_size = hyperparams['patch_size']
        self.ignored_labels = set(hyperparams['ignored_labels'])
        self.flip_augmentation = hyperparams['flip_augmentation']
        self.radiation_augmentation = hyperparams['radiation_augmentation']
        self.mixture_augmentation = hyperparams['mixture_augmentation']
        self.center_pixel = hyperparams['center_pixel']
        supervision = hyperparams['supervision']
        # Fully supervised : use all pixels with label not ignored
        if supervision == 'full':
            mask = np.ones_like(gt)
            for l in self.ignored_labels:
                mask[gt == l] = 0
        # Semi-supervised : use all pixels, except padding
        elif supervision == 'semi':
            mask = np.ones_like(gt)
        x_pos, y_pos = np.nonzero(mask)  # 提取非零值的坐标
        p = self.patch_size // 2
        self.indices = np.array([(x, y) for x, y in zip(x_pos, y_pos) if
                                 x > p and x < data.shape[0] - p and y > p and y < data.shape[1] - p])

        self.labels = [self.label[x, y] for x, y in self.indices]

        # state = np.random.get_state()
        # np.random.shuffle(self.indices)
        # np.random.set_state(state)
        # np.random.shuffle(self.labels)

    @staticmethod
    def flip(*arrays):  # 对输入的多个数组arrays进行水平或垂直翻转。
        horizontal = np.random.random() > 0.5
        vertical = np.random.random() > 0.5
        if horizontal:
            arrays = [np.fliplr(arr) for arr in arrays]
        if vertical:
            arrays = [np.flipud(arr) for arr in arrays]
        return arrays

    @staticmethod
    def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1 / 25):
        alpha = np.random.uniform(*alpha_range)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        return alpha * data + beta * noise

    def mixture_noise(self, data, label, beta=1 / 25):
        alpha1, alpha2 = np.random.uniform(0.01, 1., size=2)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        data2 = np.zeros_like(data)
        for idx, value in np.ndenumerate(label):
            if value not in self.ignored_labels:
                l_indices = np.nonzero(self.labels == value)[0]
                l_indice = np.random.choice(l_indices)
                assert (self.labels[l_indice] == value)
                x, y = self.indices[l_indice]
                data2[idx] = self.data[x, y]
        return (alpha1 * data + alpha2 * data2) / (alpha1 + alpha2) + beta * noise

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        # 得到以指定位置i为中心，size为patch_size*patch_size的图像块block
        x, y = self.indices[i]
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size

        data = self.data[x1:x2, y1:y2]
        label = self.label[x1:x2, y1:y2]

        # 数据增强
        if self.flip_augmentation and self.patch_size > 1 and np.random.random() < 0.5:
            # Perform data augmentation (only on 2D patches)
            data, label = self.flip(data, label)
        if self.radiation_augmentation and np.random.random() < 0.5:
            data = self.radiation_noise(data)
        if self.mixture_augmentation and np.random.random() < 0.5:
            data = self.mixture_noise(data, label)

        # Copy the data into numpy arrays (PyTorch doesn't like numpy views)
        data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype='float32')  # [W,H,C]->[C,W,H]
        label = np.asarray(np.copy(label), dtype='int64')

        # Load the data into PyTorch tensors
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)
        # Extract the center label if needed
        if self.center_pixel and self.patch_size > 1:
            label = label[self.patch_size // 2, self.patch_size // 2]
        # Remove unused dimensions when we work with invidual spectrums
        elif self.patch_size == 1:
            data = data[:, 0, 0]
            label = label[0, 0]
        else:
            label = self.labels[i]

        # Add a fourth dimension for 3D CNN
        # if self.patch_size > 1:
        #     # Make 4D data ((Batch x) Planes x Channels x Width x Height)
        #     data = data.unsqueeze(0)
        # plt.imshow(data[[10,23,23],:,:].permute(1,2,0))
        # plt.show()
        return data, label

class HyperGraph(torch.utils.data.Dataset):
    """ Generic class for a hyperspectral scene """

    def __init__(self, data, gt, transform=None, **hyperparams):
        """
        Args:
            data: 3D hyperspectral image
            gt: 2D array of labels
            patch_size: int, size of the spatial neighbourhood
            center_pixel: bool, set to True to consider only the label of the
                          center pixel
            data_augmentation: bool, set to True to perform random flips
            supervision: 'full' or 'semi' supervised algorithms
        """
        super(HyperGraph, self).__init__()
        self.transform = transform
        self.data = data
        self.label = gt
        self.patch_size = hyperparams['patch_size']
        self.ignored_labels = set(hyperparams['ignored_labels'])
        self.flip_augmentation = hyperparams['flip_augmentation']
        self.radiation_augmentation = hyperparams['radiation_augmentation']
        self.mixture_augmentation = hyperparams['mixture_augmentation']
        self.center_pixel = hyperparams['center_pixel']
        supervision = hyperparams['supervision']
        # Fully supervised : use all pixels with label not ignored
        if supervision == 'full':
            mask = np.ones_like(gt)
            for l in self.ignored_labels:
                mask[gt == l] = 0
        # Semi-supervised : use all pixels, except padding
        elif supervision == 'semi':
            mask = np.ones_like(gt)
        x_pos, y_pos = np.nonzero(mask)  # 提取非零值的坐标
        p = self.patch_size // 2
        self.indices = np.array([(x, y) for x, y in zip(x_pos, y_pos) if
                                 x > p and x < data.shape[0] - p and y > p and y < data.shape[1] - p])

        self.labels = [self.label[x, y] for x, y in self.indices]

        # state = np.random.get_state()
        # np.random.shuffle(self.indices)
        # np.random.set_state(state)
        # np.random.shuffle(self.labels)

    @staticmethod
    def flip(*arrays):  # 对输入的多个数组arrays进行水平或垂直翻转。
        horizontal = np.random.random() > 0.5
        vertical = np.random.random() > 0.5
        if horizontal:
            arrays = [np.fliplr(arr) for arr in arrays]
        if vertical:
            arrays = [np.flipud(arr) for arr in arrays]
        return arrays

    @staticmethod
    def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1 / 25):
        alpha = np.random.uniform(*alpha_range)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        return alpha * data + beta * noise

    def mixture_noise(self, data, label, beta=1 / 25):
        alpha1, alpha2 = np.random.uniform(0.01, 1., size=2)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        data2 = np.zeros_like(data)
        for idx, value in np.ndenumerate(label):
            if value not in self.ignored_labels:
                l_indices = np.nonzero(self.labels == value)[0]
                l_indice = np.random.choice(l_indices)
                assert (self.labels[l_indice] == value)
                x, y = self.indices[l_indice]
                data2[idx] = self.data[x, y]
        return (alpha1 * data + alpha2 * data2) / (alpha1 + alpha2) + beta * noise

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        # 得到以指定位置i为中心，size为patch_size*patch_size的图像块block
        x, y = self.indices[i]
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size

        data = self.data[x1:x2, y1:y2]
        label = self.label[x1:x2, y1:y2]

        # 数据增强
        if self.flip_augmentation and self.patch_size > 1 and np.random.random() < 0.5:
            # Perform data augmentation (only on 2D patches)
            data, label = self.flip(data, label)
        if self.radiation_augmentation and np.random.random() < 0.5:
            data = self.radiation_noise(data)
        if self.mixture_augmentation and np.random.random() < 0.5:
            data = self.mixture_noise(data, label)

        # Copy the data into numpy arrays (PyTorch doesn't like numpy views)
        data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype='float32')  # [W,H,C]->[C,W,H]
        label = np.asarray(np.copy(label), dtype='int64')

        # Load the data into PyTorch tensors
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)
        # Extract the center label if needed
        if self.center_pixel and self.patch_size > 1:
            label = label[self.patch_size // 2, self.patch_size // 2]
        # Remove unused dimensions when we work with invidual spectrums
        elif self.patch_size == 1:
            data = data[:, 0, 0]
            label = label[0, 0]
        else:
            label = self.labels[i]

        # Add a fourth dimension for 3D CNN
        # if self.patch_size > 1:
        #     # Make 4D data ((Batch x) Planes x Channels x Width x Height)
        #     data = data.unsqueeze(0)
        # plt.imshow(data[[10,23,23],:,:].permute(1,2,0))
        # plt.show()


        return data, label

def create_dataloader(args, DEVICE):
    img_src, gt_src, LABEL_VALUES_src, label_true_queue, label_false_queue, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(args.source_name,
                                                                                                     args.data_path)
    img_tar, gt_tar, LABEL_VALUES_tar, _, _, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(args.target_name,
                                                                                                     args.data_path)
    num_classes = gt_src.max()
    N_BANDS = img_src.shape[-1]
    hyperparams = vars(args)
    hyperparams.update({'n_classes': num_classes, 'n_bands': N_BANDS, 'ignored_labels': IGNORED_LABELS,
                        'device': DEVICE, 'center_pixel': False, 'supervision': 'full'})
    hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)
    r = int(hyperparams['patch_size'] / 2) + 1
    img_src = np.pad(img_src, ((r, r), (r, r), (0, 0)), 'symmetric')
    img_tar = np.pad(img_tar, ((r, r), (r, r), (0, 0)), 'symmetric')
    gt_src = np.pad(gt_src, ((r, r), (r, r)), 'constant', constant_values=(0, 0))
    gt_tar = np.pad(gt_tar, ((r, r), (r, r)), 'constant', constant_values=(0, 0))
    train_gt_src, _, training_set, _ = sample_gt(gt_src, args.training_sample_ratio, mode='random')
    test_gt_tar, _, tesing_set, _ = sample_gt(gt_tar, 1, mode='random')
    img_src_con, train_gt_src_con = img_src, train_gt_src
    for i in range(args.re_ratio - 1):
        img_src_con = np.concatenate((img_src_con, img_src))
        train_gt_src_con = np.concatenate((train_gt_src_con, train_gt_src))
    hyperparams_train = hyperparams.copy()
    if args.dataset == 'Houston':
        hyperparams_train.update(
            {'flip_augmentation': True, 'radiation_augmentation': True, 'mixture_augmentation': False})
    train_dataset = HyperX(img_src_con, train_gt_src_con, **hyperparams_train)
    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=hyperparams['batch_size'],
                                   pin_memory=True,
                                   worker_init_fn=seed_worker,
                                   generator=g,
                                   shuffle=True)
    test_dataset = HyperX(img_tar, test_gt_tar, **hyperparams)
    test_loader = data.DataLoader(test_dataset,
                                  pin_memory=True,
                                  batch_size=hyperparams['batch_size'])

    return train_loader, test_loader, IGNORED_LABELS, LABEL_VALUES_tar, label_true_queue, label_false_queue


def create_graphdataloader(args, DEVICE):
    img_src, gt_src, LABEL_VALUES_src, _, _, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(
        args.source_name,
        args.data_path)
    img_tar, gt_tar, LABEL_VALUES_tar, _, _, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(args.target_name,
                                                                                              args.data_path)
    num_classes = gt_src.max()
    N_BANDS = img_src.shape[-1]
    hyperparams = vars(args)
    hyperparams.update({'n_classes': num_classes, 'n_bands': N_BANDS, 'ignored_labels': IGNORED_LABELS,
                        'device': DEVICE, 'center_pixel': False, 'supervision': 'full'})
    hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)
    r = int(hyperparams['patch_size'] / 2) + 1
    img_src = np.pad(img_src, ((r, r), (r, r), (0, 0)), 'symmetric')
    img_tar = np.pad(img_tar, ((r, r), (r, r), (0, 0)), 'symmetric')
    gt_src = np.pad(gt_src, ((r, r), (r, r)), 'constant', constant_values=(0, 0))
    gt_tar = np.pad(gt_tar, ((r, r), (r, r)), 'constant', constant_values=(0, 0))
    train_gt_src, _, training_set, _ = sample_gt(gt_src, args.training_sample_ratio, mode='random')
    test_gt_tar, _, tesing_set, _ = sample_gt(gt_tar, 1, mode='random')
    img_src_con, train_gt_src_con = img_src, train_gt_src
    for i in range(args.re_ratio - 1):
        img_src_con = np.concatenate((img_src_con, img_src))
        train_gt_src_con = np.concatenate((train_gt_src_con, train_gt_src))

    print(img_src_con.shape, train_gt_src_con.shape)
    print(img_tar.shape, test_gt_tar.shape)

    non_zero_train = np.argwhere(train_gt_src_con != 0)
    non_zero_test = np.argwhere(test_gt_tar != 0)
    train_patches, train_patches_gt = [], []
    test_patches, test_patches_gt = [], []
    for index in non_zero_train:
        y, x = index  # 获取当前像素的坐标
        half_height, half_width = args.patch_size // 2, args.patch_size // 2
        patch = img_src_con[y - half_height-1:y + half_height, x - half_width-1:x + half_width, :]
        train_patches.append(patch)
        train_patches_gt.append(train_gt_src_con[y, x])
    train_patches = np.array(train_patches)
    train_patches_gt = np.array(train_patches_gt)
    for index in non_zero_test:
        y, x = index  # 获取当前像素的坐标
        half_height, half_width = args.patch_size // 2, args.patch_size // 2
        patch = img_tar[y - half_height-1:y + half_height, x - half_width-1:x + half_width, :]
        test_patches.append(patch)
        test_patches_gt.append(test_gt_tar[y, x])
    test_patches = np.array(test_patches)
    test_patches_gt = np.array(test_patches_gt)
    indexs_train = torch.zeros((train_patches.shape[0], args.n_gcn), dtype=int)
    indexs_test = torch.zeros((test_patches.shape[0], args.n_gcn), dtype=int)
    for i in range(train_patches.shape[0]):
        indexs_train[i] = gain_neighborhood_pixel(train_patches[i], args)
    for i in range(test_patches.shape[0]):
        indexs_test[i] = gain_neighborhood_pixel(test_patches[i], args)
    print(train_patches.shape, train_patches_gt.shape, indexs_train.shape)
    print(test_patches.shape, test_patches_gt.shape, indexs_test.shape)
    train_argmentation_patches = []
    if args.dataset == 'Houston' or args.dataset == 'HyRANK':
        for patch in train_patches:
            augmented_patch = augment_data(patch, 1, 1, 0)
            train_argmentation_patches.append(augmented_patch)
    else:
        train_argmentation_patches = train_patches

    train_argmentation_patches = np.array(train_argmentation_patches)
    print(train_argmentation_patches.shape)
    A_train = GET_A2(data=torch.from_numpy(train_argmentation_patches).type(torch.FloatTensor), l=3, sigma=10)
    # A_train = GET_A2(data=train_argmentation_patches, l=3, sigma=10)
    train_argmentation_patches = torch.from_numpy(train_argmentation_patches).type(torch.FloatTensor)  # [695, 200, 7, 7]
    train_patches_gt = torch.from_numpy(train_patches_gt).type(torch.LongTensor)  # [695]
    # index_edges_train = get_index_edges(A_train)
    print(A_train.shape, train_argmentation_patches.shape, train_patches_gt.shape)
    A_test = GET_A2(data=torch.from_numpy(test_patches).type(torch.FloatTensor), l=3, sigma=10)
    # A_test = GET_A2(test_patches, l=3, sigma=10)
    test_patches = torch.from_numpy(test_patches).type(torch.FloatTensor)  # [695, 200, 7, 7]
    test_patches_gt = torch.from_numpy(test_patches_gt).type(torch.LongTensor)  # [695]
    # index_edges_test = get_index_edges(A_test)
    print(A_test.shape, test_patches.shape, test_patches_gt.shape)
    # print(A_train)
    # label_train_dataset = data.TensorDataset(A_train, train_argmentation_patches, train_patches_gt)
    # label_test_dataset = data.TensorDataset(A_test, test_patches, test_patches_gt)

    label_train_dataset = data.TensorDataset(A_train, train_argmentation_patches, train_patches_gt)
    label_test_dataset = data.TensorDataset(A_test, test_patches, test_patches_gt)
    g = torch.Generator()
    g.manual_seed(args.seed)
    label_train_loader = data.DataLoader(label_train_dataset,
                                         batch_size=args.batch_size,
                                         worker_init_fn=seed_worker,
                                         generator=g,
                                         pin_memory=True,
                                         shuffle=True)
    label_test_loader = data.DataLoader(label_test_dataset,
                                        batch_size=args.batch_size,
                                        pin_memory=True,
                                        shuffle=False)
    return label_train_loader, label_test_loader, IGNORED_LABELS, LABEL_VALUES_tar, indexs_train, indexs_test