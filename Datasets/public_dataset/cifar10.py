import torch
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from Datasets.public_dataset.utils.public_dataset import PublicDataset, GaussianBlur
from Datasets.utils.transforms import DeNormalize
from utils.conf import single_domain_data_path
from PIL import Image
from typing import Tuple
import torchvision.transforms as T

class MyCIFAR10(CIFAR10):
    """
    Overrides the CIFAR10 dataset to change the getitem function.
    """

    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=True, data_name=None) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.data_name = data_name
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.dataset = self.__build_truncated_dataset__()
        self.data = self.dataset.data

        #print(self.dataset.__dir__())
        #print(self.dataset)

        #['root', 'transform', 'target_transform', 'transforms', 'train', 'data', 'targets', 'classes', 'class_to_idx', '__module__', '__doc__', 'base_folder', 'url', 'filename', 'tgz_md5', 'train_list', 'test_list', 'meta', '__init__', '_load_meta', '__getitem__', '__len__', '_check_integrity', 'download', 'extra_repr', '__parameters__', '__slotnames__', '_repr_indent', '__repr__', '_format_transform_repr', '__add__', '__orig_bases__', '__dict__', '__weakref__', '__class_getitem__', '__init_subclass__', '__new__', '__hash__', '__str__', '__getattribute__', '__setattr__', '__delattr__', '__lt__', '__le__', '__eq__', '__ne__', '__gt__', '__ge__', '__reduce_ex__', '__reduce__', '__getstate__', '__subclasshook__', '__format__', '__sizeof__', '__dir__', '__class__']

        if hasattr(self.dataset, 'classes'):
            self.classes = self.dataset.classes

        if hasattr(self.dataset, 'labels'):
            self.targets = self.dataset.labels

        elif hasattr(self.dataset, 'targets'):
            self.targets = self.dataset.targets

        if isinstance(self.targets, torch.Tensor):
            self.targets = self.targets.numpy()
        if isinstance(self.data, torch.Tensor):
            self.data = self.data.numpy()

    def __build_truncated_dataset__(self):
        dataobj = CIFAR10('~/miniconda3/lib/python3.12/site-packages/torchvision/datasets$/cifar10', self.train, self.transform, self.target_transform, self.download)

        return dataobj

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[type(Image), int, type(Image)]:
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img, mode='RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


class PublicCIFAR10(PublicDataset):
    NAME = 'pub_cifar10'

    def __init__(self, args, cfg, **kwargs) -> None:
        super().__init__(args, cfg, **kwargs)

        self.strong_aug = transforms.Compose([
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor()])

        self.weak_aug = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.ToTensor()])

        self.pub_len=kwargs['pub_len']
        self.public_batch_size=kwargs['public_batch_size']
        self.aug=kwargs['pub_aug']

        '''

        normalization = self.get_normalization_transform()

        self.weak_transform = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             normalization])

        self.strong_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalization])

        '''


    def get_data_loaders(self):
        #train_dataset = MyCIFAR10('~/miniconda3/lib/python3.12/site-packages/torchvision/datasets$/cifar10/cifar-10-batches-py/data_batch_1', train=True,
        #                          download=True, transform=train_transform)
        #test_transform = transforms.Compose(
        #    [transforms.ToTensor(), self.get_normalization_transform()])
        #test_dataset = CIFAR10('~/miniconda3/lib/python3.12/site-packages/torchvision/datasets$/cifar10/cifar-10-batches-py/test', train=False,
        #                       download=True, transform=test_transform)

        if self.aug == 'two_weak':
            train_transform = TwoCropsTransform(self.weak_aug, self.weak_aug)

        elif self.aug == 'two_strong':
            train_transform = TwoCropsTransform(self.strong_aug, self.strong_aug)

        else:
            train_transform = self.weak_aug

        train_dataset = MyCIFAR10(data_name="cifar10", root=single_domain_data_path(),
                               transform=train_transform)

        self.traindl = self.random_loaders(train_dataset, self.pub_len, self.public_batch_size)