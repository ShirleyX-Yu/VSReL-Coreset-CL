# coding:utf8-*-

import torchvision
from torch.utils.data import Dataset, DataLoader, Subset

from dataset import idataset


class IDMergeDataset(Dataset):
    def __init__(self, data, transforms, id_bias=0):
        self.data = []
        self.id_bias = id_bias
        for i, di in enumerate(data):
            self.data.append([i + self.id_bias] + di)
        self.transforms = transforms
        self.to_tensor = torchvision.transforms.ToTensor()

    def __getitem__(self, index):
        di = self.data[index]
        d_id, sp, lab = di
        aug_sp = self.transforms(sp)
        ori_sp = self.to_tensor(sp)
        return d_id, ori_sp, aug_sp, lab

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for idx, di in enumerate(self.data):
            d_id, sp, lab = di
            aug_sp = self.transforms(sp)
            ori_sp = self.to_tensor(sp)
            yield d_id, ori_sp, aug_sp, lab


def get_data_loaders(dataset, data_path, batch_size, use_bn):
    # add data id to data
    train_loaders = []
    test_loaders = []
    to_pil = torchvision.transforms.ToPILImage()
    if dataset == 'splittinyimagenet':
        generator = idataset.SplitTinyImageNet(
            data_path=data_path, batch_size=batch_size)
        transforms = generator.get_transforms()
        id_bias = 0
        for i in range(generator.max_iter):
            _, task_slt_loader, task_test_loader = generator.next_task()
            train_data = []
            for di in task_slt_loader.dataset:
                sp, lab = di
                pil_sp = to_pil(sp)
                train_data.append([pil_sp, lab])
            task_train_dataset = IDMergeDataset(
                data=train_data,
                transforms=transforms,
                id_bias=id_bias
            )
            task_train_loader = DataLoader(
                task_train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
            id_bias += len(train_data)
            train_loaders.append(task_train_loader)
            test_loaders.append(task_test_loader)
        model_params = {
            'model_type': 'resnet',
            'num_class': 200,
            'num_blocks': [2, 2, 2, 2],
            'use_bn': use_bn,
            'setting': 'der'
        }
    elif dataset == 'splitcifar':
        generator = idataset.SplitCifar(
            aug_type='der', limit_per_task=10000)
        transforms = generator.get_transforms()
        id_bias = 0
        for i in range(generator.max_iter):
            train_inds, full_train_inds, test_inds = generator.next_task()
            slt_dataset = Subset(generator.train_dataset_wo_augment, train_inds)
            train_data = []
            for di in slt_dataset:
                sp, lab = di
                pil_sp = to_pil(sp)
                train_data.append([pil_sp, lab])
            task_train_dataset = IDMergeDataset(
                data=train_data,
                transforms=transforms,
                id_bias=id_bias
            )
            task_train_loder = DataLoader(task_train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
            id_bias += len(train_data)
            train_loaders.append(task_train_loder)
            test_loaders.append(
                idataset.get_custom_loader(
                    generator.test_dataset, test_inds, batch_size=100)
            )
        model_params = {
            'model_type': 'resnet',
            'num_class': 10,
            'num_blocks': [2, 2, 2, 2],
            'use_bn': use_bn,
            'setting': 'der'
        }
    elif dataset == 'splitcifar100':
        generator = idataset.SplitCifar100(
            limit_per_task=5000, data_path=data_path)
        transforms = generator.get_transforms()
        id_bias = 0
        for i in range(generator.max_iter):
            train_inds, full_train_inds, test_inds = generator.next_task()
            slt_dataset = Subset(generator.train_dataset_wo_augment, train_inds)
            train_data = []
            for di in slt_dataset:
                sp, lab = di
                pil_sp = to_pil(sp)
                train_data.append([pil_sp, lab])
            task_train_dataset = IDMergeDataset(
                data=train_data,
                transforms=transforms,
                id_bias=id_bias
            )
            task_train_loder = DataLoader(task_train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
            id_bias += len(train_data)
            train_loaders.append(task_train_loder)
            test_loaders.append(
                idataset.get_custom_loader(
                    generator.test_dataset, test_inds, batch_size=100)
            )
        model_params = {
            'model_type': 'resnet',
            'num_class': 100,
            'num_blocks': [2, 2, 2, 2],
            'use_bn': bool(use_bn),
            'setting': 'der'
        }
    else:
        raise ValueError('Invalid dataset')
    return train_loaders, test_loaders, model_params, transforms
