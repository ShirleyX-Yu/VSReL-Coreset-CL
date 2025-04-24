# -*-coding:utf8-*-

import os
import numpy as np
import torchvision.transforms
from torchvision import datasets
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import copy
from PIL import Image

from dataset.single_task_dataset import MergeDataset, SimpleDataset


class SplitTinyImageNet(object):
    def __init__(self, data_path, batch_size):
        self.data_path = data_path
        self.batch_size = batch_size
        # process data to PIL image, load both train and eval dataset
        self.train_data = []
        train_sps = []
        train_labs = []
        for i in range(20):
            sp = np.load(
                os.path.join(self.data_path, 'processed/x_train_%02d.npy' % (i + 1)))
            train_sps.append(sp)
            lab = np.load(
                os.path.join(self.data_path, 'processed/y_train_%02d.npy' % (i + 1)))
            train_labs.append(lab)
        train_sps = np.concatenate(train_sps, axis=0)
        train_labs = np.concatenate(train_labs, axis=0)
        for i in range(train_sps.shape[0]):
            np_img = train_sps[i, :]
            img = Image.fromarray(np.uint8(255 * np_img))
            lab = int(train_labs[i])
            self.train_data.append([img, lab])
        del train_sps
        del train_labs
        self.eval_data = []
        eval_sps = []
        eval_labs = []
        for i in range(20):
            sp = np.load(
                os.path.join(self.data_path, 'processed/x_val_%02d.npy' % (i + 1)))
            eval_sps.append(sp)
            lab = np.load(
                os.path.join(self.data_path, 'processed/y_val_%02d.npy' % (i + 1)))
            eval_labs.append(lab)
        eval_sps = np.concatenate(eval_sps, axis=0)
        eval_labs = np.concatenate(eval_labs, axis=0)
        for i in range(eval_sps.shape[0]):
            np_img = eval_sps[i, :]
            img = Image.fromarray(np.uint8(255 * np_img))
            lab = int(eval_labs[i])
            self.eval_data.append([img, lab])

        # make transforms
        self.transform_train = torchvision.transforms.Compose(
            [torchvision.transforms.RandomCrop(64, padding=4),
             torchvision.transforms.RandomHorizontalFlip(),
             torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize((0.4802, 0.4480, 0.3975), (0.2770, 0.2691, 0.2821))]
        )
        self.transform_test = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize((0.4802, 0.4480, 0.3975), (0.2770, 0.2691, 0.2821))]
        )
        self.to_tensor = torchvision.transforms.ToTensor()

        self.task_dic = {}
        self.task_dic = self.make_task_dic()
        self.max_iter = len(self.task_dic)
        self.cur_iter = 0

        # make data loaders
        self.train_loaders = []
        self.slt_loaders = []
        self.test_loaders = []
        for i in range(self.max_iter):
            train_loader, slt_loader, test_loader = self.make_dataset(task_id=i)
            self.train_loaders.append(train_loader)
            self.slt_loaders.append(slt_loader)
            self.test_loaders.append(test_loader)

    def make_dataset(self, task_id):
        # make both train dataset and test dataset
        class_set = set(self.task_dic[task_id])
        task_train_data = []
        for di in self.train_data:
            sp, lab = di
            if lab in class_set:
                task_train_data.append(di)
        task_train_dataset = MergeDataset(
            data=task_train_data,
            transforms=self.transform_train
        )
        train_loader = DataLoader(
            task_train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)
        task_slt_dataset = SimpleDataset(
            data=task_train_data,
            transforms=self.to_tensor
        )
        slt_loader = DataLoader(
            task_slt_dataset, batch_size=len(task_train_data), shuffle=False, drop_last=False)
        task_test_data = []
        for di in self.eval_data:
            sp, lab = di
            if lab in class_set:
                task_test_data.append(di)
        task_test_dataset = SimpleDataset(
            data=task_test_data,
            transforms=self.transform_test
        )
        test_loader = DataLoader(
            task_test_dataset, batch_size=100, shuffle=False, drop_last=False)
        return train_loader, slt_loader, test_loader

    def make_task_dic(self):
        tasks = 10
        cls_per_task = 20
        cur_class = 0
        for i in range(tasks):
            self.task_dic[i] = []
            for j in range(cls_per_task):
                self.task_dic[i].append(cur_class)
                cur_class += 1
        return self.task_dic

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise Exception('Number of tasks exceeded!')
        else:
            self.cur_iter += 1
            return self.train_loaders[self.cur_iter - 1], self.slt_loaders[self.cur_iter - 1],\
                self.test_loaders[self.cur_iter - 1]

    def get_transforms(self):
        return self.transform_train

    def get_task_dic(self):
        return self.task_dic

    def get_eval_transforms(self):
        return self.transform_test


class SplitCifar100(object):
    def __init__(self, limit_per_task, data_path=''):
        self.current_pos = 0
        self.transform_train = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        self.transform_test = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))]
        )

        self.to_tensor = torchvision.transforms.ToTensor()
        self.data_path = '../data/CIFAR100/'
        if data_path != '':
            self.data_path = data_path
        self.train_dataset = torchvision.datasets.CIFAR100(
            self.data_path, train=True, transform=self.transform_train, download=True)
        self.train_dataset_wo_augment = torchvision.datasets.CIFAR100(
            self.data_path, train=True, transform=self.to_tensor, download=False)
        self.test_dataset = torchvision.datasets.CIFAR100(
            self.data_path, train=False, transform=self.transform_test, download=False)

        self.Y_train = np.array(self.train_dataset.targets)
        self.Y_test = np.array(self.test_dataset.targets)

        self.task_dic = {}
        self.task_dic = self.make_task_dic()
        self.max_iter = len(self.task_dic)
        self.cur_iter = 0

        self.limit_per_task = limit_per_task

        self.inds = []
        self.full_inds = []
        rs = np.random.RandomState(0)
        for i in range(len(self.task_dic)):
            cur_inds = []
            for ci in self.task_dic[i]:
                inds = np.where(self.Y_train == ci)[0]
                cur_inds.append(inds)
            ind = np.concatenate(cur_inds, axis=0)
            self.full_inds.append(ind)
            sub_ind = rs.choice(ind, limit_per_task, replace=False)
            self.inds.append(sub_ind)

        self.all_inds = np.hstack(self.inds)

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise Exception('Number of tasks exceeded!')
        else:
            # Retrieve train data
            train_ind = self.inds[self.cur_iter]
            full_train_id = self.full_inds[self.cur_iter]
            # Retrieve test data
            test_ind = []
            for cls in self.task_dic[self.cur_iter]:
                cls_ind = np.where(self.Y_test == cls)[0]
                test_ind.append(cls_ind)
            test_ind = np.concatenate(test_ind, axis=0)
            self.cur_iter += 1
            return train_ind, full_train_id, test_ind

    def make_task_dic(self):
        tasks = 10
        cls_per_task = 10
        cur_class = 0
        for i in range(tasks):
            self.task_dic[i] = []
            for j in range(cls_per_task):
                self.task_dic[i].append(cur_class)
                cur_class += 1
        return self.task_dic

    def get_transforms(self):
        return self.transform_train

    def get_task_dic(self):
        return self.task_dic

    def get_eval_transforms(self):
        return self.transform_test


class SplitCifar(object):
    def __init__(self, limit_per_task=None, aug_type='greedy', data_path=''):
        self.current_pos = 0
        if limit_per_task is None:
            limit_per_task = 1000
        if aug_type == 'greedy':
            self.transform_train = torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            self.transform_test = torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor(),
                 torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
            )
        elif aug_type == 'der':
            self.transform_train = torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2615))
            ])
            self.transform_test = torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor(),
                 torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2615))]
            )
        else:
            raise ValueError('Invalid runner type')
        self.to_tensor = torchvision.transforms.ToTensor()
        self.data_path = '../data/CIFAR10/'
        if len(data_path) > 0:
            self.data_path = data_path

        self.train_dataset = datasets.CIFAR10(
            self.data_path, train=True, transform=self.transform_train, download=True)
        self.train_dataset_wo_augment = datasets.CIFAR10(
            self.data_path, train=True, transform=self.to_tensor, download=False)
        self.test_dataset = datasets.CIFAR10(
            self.data_path, train=False, transform=self.transform_test, download=False)

        self.Y_train = np.array(self.train_dataset.targets)
        self.Y_test = np.array(self.test_dataset.targets)

        self.sets_0 = [0, 2, 4, 6, 8]
        self.sets_1 = [1, 3, 5, 7, 9]
        self.max_iter = len(self.sets_0)
        self.cur_iter = 0
        self.limit_per_task = limit_per_task

        self.inds = []
        self.full_inds = []
        rs = np.random.RandomState(seed=0)
        for i in range(5):
            ind = np.where(np.logical_or(self.Y_train == self.sets_0[i], self.Y_train == self.sets_1[i]))[0]
            self.full_inds.append(ind)
            ind = rs.choice(ind, self.limit_per_task, replace=False)
            self.inds.append(ind)
        self.all_inds = np.hstack(self.inds)

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise Exception('Number of tasks exceeded!')
        else:
            # Retrieve train data
            train_ind = self.inds[self.cur_iter]
            full_train_ind = self.full_inds[self.cur_iter]
            # Retrieve test data
            test_ind = np.where(
                np.logical_or(self.Y_test == self.sets_0[self.cur_iter], self.Y_test == self.sets_1[self.cur_iter]))[0]
            self.cur_iter += 1
            return train_ind, full_train_ind, test_ind

    def get_transforms(self):
        return self.transform_train

    def get_task_dic(self):
        task_dic = {}
        for i in range(len(self.sets_0)):
            task_dic[i] = [self.sets_0[i], self.sets_1[i]]
        return task_dic

    def get_eval_transforms(self):
        return self.transform_test


class SplitMNIST(object):
    def __init__(self, imbalanced=True):
        self.current_pos = 0
        if imbalanced:
            limit_per_task = 200
        else:
            limit_per_task = 1000
        self.transform_train = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.transform_test = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.to_tensor = torchvision.transforms.ToTensor()

        self.train_dataset = datasets.MNIST(
            '../data/MNIST/', train=True, transform=self.transform_train, download=True)
        self.train_dataset_wo_augment = datasets.MNIST(
            '../data/MNIST/', train=True, transform=self.to_tensor, download=False)
        self.test_dataset = datasets.MNIST(
            '../data/MNIST/', train=False, transform=self.transform_test, download=False)

        self.Y_train = np.array(self.train_dataset.targets)
        self.Y_test = np.array(self.test_dataset.targets)

        self.sets_0 = [0, 2, 4, 6, 8]
        self.sets_1 = [1, 3, 5, 7, 9]
        self.max_iter = len(self.sets_0)
        self.cur_iter = 0
        self.limit_per_task = limit_per_task

        self.inds = []
        self.full_inds = []
        rs = np.random.RandomState(seed=0)
        for i in range(5):
            if i == 4 and imbalanced:
                limit_per_task = 2000
            ind = np.where(np.logical_or(self.Y_train == self.sets_0[i], self.Y_train == self.sets_1[i]))[0]
            self.full_inds.append(ind)
            ind = rs.choice(ind, limit_per_task, replace=False)
            self.inds.append(ind)
        self.all_inds = np.hstack(self.inds)

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise Exception('Number of tasks exceeded!')
        else:
            # Retrieve train data
            train_ind = self.inds[self.cur_iter]
            full_train_ind = self.full_inds[self.cur_iter]
            # Retrieve test data
            test_ind = np.where(
                np.logical_or(self.Y_test == self.sets_0[self.cur_iter], self.Y_test == self.sets_1[self.cur_iter]))[0]
            self.cur_iter += 1
            return train_ind, full_train_ind, test_ind

    def get_transforms(self):
        return self.transform_train

    def get_task_dic(self):
        task_dic = {}
        for i in range(len(self.sets_0)):
            task_dic[i] = [self.sets_0[i], self.sets_1[i]]
        return task_dic

    def get_eval_transforms(self):
        return self.transform_test


class PermutedMnistGenerator(object):
    def __init__(self, limit_per_task=1000, max_iter=10):
        self.transform_train = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.transform_test = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.X_train_batch = []
        self.y_train_batch = []
        self.current_pos = 0
        self.cur_iter = 0
        self.to_tensor = torchvision.transforms.ToTensor()

        train_dataset = datasets.MNIST(
            '../data/MNIST/', train=True, transform=self.transform_train, download=True)
        train_dataset_wo_augment = datasets.MNIST(
            '../data/MNIST/', train=True, transform=self.to_tensor, download=False)
        test_dataset = datasets.MNIST(
            '../data/MNIST/', train=False, transform=self.transform_test, download=True)

        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
        train_loader_wo_aug = DataLoader(train_dataset, batch_size=len(train_dataset_wo_augment), shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

        self.X_train, self.Y_train = next(iter(train_loader))
        self.X_train, self.Y_train = self.X_train.numpy()[:limit_per_task].reshape(-1, 28 * 28), self.Y_train.numpy()[
                                                                                                 :limit_per_task]
        self.X_train_wo_aug, self.Y_train_wo_aug = next(iter(train_loader_wo_aug))
        self.X_train_wo_aug = self.X_train_wo_aug.numpy().reshape(-1, 28 * 28)
        self.Y_train_wo_aug = self.Y_train_wo_aug.numpy()
        self.X_train_wo_aug_sub = self.X_train_wo_aug[:limit_per_task]
        self.Y_train_wo_aug_sub = self.Y_train_wo_aug[:limit_per_task]
        self.X_test, self.Y_test = next(iter(test_loader))
        self.X_test, self.Y_test = self.X_test.numpy().reshape(-1, 28 * 28), self.Y_test.numpy()

        self.max_iter = max_iter
        self.permutations = []

        self.rs = np.random.RandomState(0)

        for i in range(max_iter):
            perm_inds = list(range(self.X_train.shape[1]))
            self.rs.shuffle(perm_inds)
            self.permutations.append(perm_inds)
            self.X_train_batch.append(self.X_train[:, perm_inds])
            self.y_train_batch.append(self.Y_train)

        self.X_train_batch = np.vstack(self.X_train_batch)
        self.y_train_batch = np.hstack(self.y_train_batch)

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise Exception('Number of tasks exceeded!')
        else:
            perm_inds = self.permutations[self.cur_iter]

            next_x_train = copy.deepcopy(self.X_train)
            next_x_train = next_x_train[:, perm_inds]
            next_y_train = self.Y_train

            next_x_train_wo_aug = copy.deepcopy(self.X_train_wo_aug)
            next_x_train_wo_aug = next_x_train_wo_aug[:, perm_inds]
            next_x_train_wo_aug = next_x_train_wo_aug.reshape(-1, 1, 28, 28)
            next_y_train_wo_aug = self.Y_train_wo_aug
            next_x_train_wo_aug_sub = copy.deepcopy(self.X_train_wo_aug_sub)
            next_x_train_wo_aug_sub = next_x_train_wo_aug_sub[:, perm_inds]
            next_x_train_wo_aug_sub = next_x_train_wo_aug_sub.reshape(-1, 1, 28, 28)
            next_y_train_wo_aug_sub = self.Y_train_wo_aug_sub

            next_x_test = copy.deepcopy(self.X_test)
            next_x_test = next_x_test[:, perm_inds]
            next_y_test = self.Y_test

            self.cur_iter += 1
            return next_x_train, next_y_train, next_x_test, next_y_test, \
                [next_x_train_wo_aug, next_y_train_wo_aug, next_x_train_wo_aug_sub, next_y_train_wo_aug_sub]

    def get_transforms(self):
        return self.transform_train

    def get_task_dic(self):
        task_dic = {}
        for i in range(self.max_iter):
            task_dic[i] = list(range(10))
        return task_dic

    def get_eval_transforms(self):
        return self.transform_test


def get_custom_loader(dataset, inds, batch_size, shuffle=True):
    data_loader = DataLoader(
        Subset(dataset, inds),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True
    )
    return data_loader


def merge_custom_loader(trainset_no_aug, transforms, batch_size, shuffle=True):
    data = []
    to_pil = torchvision.transforms.ToPILImage()
    for di in trainset_no_aug:
        sp, lab = di
        data.append([to_pil(sp), lab])
    dataset = MergeDataset(
        data=data,
        transforms=transforms
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=False,
        shuffle=shuffle
    )
    return data_loader


def get_dataset(opts):
    train_loaders = []
    train_sub_loaders_wo_aug = []
    test_loaders = []
    if opts.dataset == 'splittinyimagenet':
        generator = SplitTinyImageNet(
            data_path=opts.data_path,
            batch_size=opts.batch_size
        )
        transforms = generator.get_transforms()
        eval_transforms = generator.get_eval_transforms()
        for i in range(generator.max_iter):
            task_train_loader, task_slt_loader, task_test_loader = generator.next_task()
            train_loaders.append(task_train_loader)
            test_loaders.append(task_test_loader)
        model_params = {
            'model_type': 'resnet',
            'num_class': 200,
            'num_blocks': [2, 2, 2, 2],
            'use_bn': bool(opts.use_bn),
            'setting': opts.setting
        }
        task_dic = generator.get_task_dic()
    elif opts.dataset == 'splitcifar':
        generator = SplitCifar(
            limit_per_task=None if opts.limit_per_task < 0 else opts.limit_per_task,
            aug_type=opts.aug_type,
            data_path=opts.data_path
        )
        transforms = generator.get_transforms()
        eval_transforms = generator.get_eval_transforms()
        for i in range(generator.max_iter):
            train_inds, full_train_inds, test_inds = generator.next_task()
            train_sub_loaders_wo_aug.append(
                get_custom_loader(
                    generator.train_dataset_wo_augment, train_inds, batch_size=len(train_inds), shuffle=False)
            )
            test_loaders.append(get_custom_loader(generator.test_dataset, test_inds, batch_size=opts.batch_size))
            if opts.runner_type == 'coreset':
                train_loaders.append(
                    get_custom_loader(generator.train_dataset, train_inds, batch_size=opts.batch_size)
                )
            else:
                raise ValueError('Invalid runner type')
        model_params = {
            'model_type': 'resnet',
            'num_class': 10,
            'num_blocks': [2, 2, 2, 2],
            'use_bn': bool(opts.use_bn),
            'setting': opts.setting
        }
        task_dic = generator.get_task_dic()
    elif opts.dataset == 'splitcifar100':
        generator = SplitCifar100(limit_per_task=opts.limit_per_task)
        transforms = generator.get_transforms()
        eval_transforms = generator.get_eval_transforms()
        for i in range(generator.max_iter):
            train_inds, full_train_inds, test_inds = generator.next_task()
            train_sub_loaders_wo_aug.append(
                get_custom_loader(
                    generator.train_dataset_wo_augment, train_inds, batch_size=len(train_inds), shuffle=False)
            )
            test_loaders.append(get_custom_loader(generator.test_dataset, test_inds, batch_size=opts.batch_size))
            if opts.runner_type == 'coreset':
                train_loaders.append(
                    get_custom_loader(generator.train_dataset, train_inds, batch_size=opts.batch_size)
                )
            else:
                raise ValueError('Invalid runner type')
        model_params = {
            'model_type': 'resnet',
            'num_class': 100,
            'num_blocks': [2, 2, 2, 2],
            'use_bn': bool(opts.use_bn),
            'setting': opts.setting
        }
        task_dic = generator.get_task_dic()
    elif opts.dataset == 'splitmnist':
        generator = SplitMNIST(imbalanced=False)
        for i in range(generator.max_iter):
            train_inds, full_train_inds, test_inds = generator.next_task()
            train_loaders.append(get_custom_loader(generator.train_dataset, train_inds, batch_size=opts.batch_size))
            train_sub_loaders_wo_aug.append(
                get_custom_loader(
                    generator.train_dataset_wo_augment, train_inds, batch_size=len(train_inds), shuffle=False)
            )
            test_loaders.append(get_custom_loader(generator.test_dataset, test_inds, batch_size=opts.batch_size))
        model_params = {
            'model_type': 'cnn',
            'num_class': 10
        }
        transforms = generator.get_transforms()
        eval_transforms = generator.get_eval_transforms()
        task_dic = generator.get_task_dic()
    else:
        raise ValueError('Invalid dataset')
    return model_params, transforms, eval_transforms, task_dic, train_loaders, train_sub_loaders_wo_aug, test_loaders
