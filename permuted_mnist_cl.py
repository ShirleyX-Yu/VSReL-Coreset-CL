# -*-coding:utf8-*-

import argparse
import os
import numpy as np
import torchvision.transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import copy

import utils
from continual_learning import continual_runner
from dataset import single_task_dataset


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


def make_selection_params(opts):
    selection_params = {
        'init_size': 0,
        'class_balance': False,
        'only_new_data': True,
        'cur_train_steps': opts.cur_train_steps,
        'cur_train_lr': opts.cur_train_lr,
        'selection_steps': opts.selection_steps,
        'ideal_logit': True,
        'logit_compute_mode': 'end_task',
        'loss_params': {
            'ce_factor': 1.0,
            'mse_factor': opts.slt_mse_factor
        },
        'ref_train_params': {
            'lr': opts.ref_train_lr,
            'epochs': opts.ref_train_epoch,
            'batch_size': 32,
            'eval_batch_size': 20,
            'use_cuda': bool(opts.use_cuda),
            'early_stop': -1,
            'log_steps': 100,
            'opt_type': 'sgd',
            'loss_params': {
                'ce_factor': 1.0,
                'mse_factor': 0.0
            },
            'ref_sample_per_task': opts.ref_sample_per_task  # default 0
        }
    }
    return selection_params


def main(opts):
    if not os.path.exists(opts.local_path):
        os.makedirs(opts.local_path)
    # make data loaders
    train_loaders = []
    train_loaders_wo_aug = []
    train_sub_loaders_wo_aug = []
    test_loaders = []
    if opts.dataset == 'permmnist':
        generator = PermutedMnistGenerator()
        for i in range(generator.max_iter):
            x_train, y_train, x_test, y_test, slt_data = generator.next_task()
            train_dataset = single_task_dataset.NumpyDataset(data=x_train, target=y_train)
            train_loaders.append(
                DataLoader(
                    train_dataset,
                    batch_size=opts.batch_size,
                    shuffle=True,
                    pin_memory=True
                )
            )
            test_dataset = single_task_dataset.NumpyDataset(data=x_test, target=y_test)
            test_loaders.append(
                DataLoader(
                    test_dataset,
                    batch_size=opts.batch_size,
                    pin_memory=True
                )
            )
            train_dataset_wo_aug = single_task_dataset.NumpyDataset(data=slt_data[0], target=slt_data[1])
            train_loaders_wo_aug.append(
                DataLoader(
                    train_dataset_wo_aug,
                    batch_size=len(train_dataset_wo_aug),
                    shuffle=False,
                    pin_memory=True
                )
            )
            train_dataset_wo_aug_sub = single_task_dataset.NumpyDataset(data=slt_data[2], target=slt_data[3])
            train_sub_loaders_wo_aug.append(
                DataLoader(
                    train_dataset_wo_aug_sub,
                    batch_size=len(train_dataset_wo_aug_sub),
                    shuffle=False,
                    pin_memory=True
                )
            )
        model_params = {
            'model_type': 'mlp',
            'input_dim': 28 * 28,
            'interm_dim': 100,
            'num_class': 10
        }
        selection_params = make_selection_params(opts=opts)
        eval_transforms = generator.get_eval_transforms()
        task_dic = generator.get_task_dic()
    else:
        raise ValueError('Invalid dataset')
    # modify selection params
    if opts.slt_mse_factor >= 0:
        selection_params['loss_params'] = {
            'ce_factor': 1.0,
            'mse_factor': opts.slt_mse_factor
        }
    if opts.cur_train_steps > 0:
        selection_params['cur_train_steps'] = opts.cur_train_steps
    if opts.ref_train_epoch > 0:
        selection_params['ref_train_params']['epochs'] = opts.ref_train_epoch
    if opts.selection_steps > 0:
        selection_params['selection_steps'] = opts.selection_steps
    if opts.ref_train_lr > 0:
        selection_params['ref_train_params']['lr'] = opts.ref_train_lr
    if opts.cur_train_lr > 0:
        selection_params['cur_train_lr'] = opts.cur_train_lr
    if opts.ref_sample_per_task > 0:
        selection_params['ref_train_params']['ref_sample_per_task'] = opts.ref_sample_per_task
    for k in selection_params.keys():
        print(k, '\t\t', selection_params[k])
    # build continual runner
    train_params = {
        'lr': opts.lr,
        'alpha': opts.alpha,
        'epochs': opts.epochs,
        'batch_size': opts.batch_size,
        'mem_batch_size': opts.mem_batch_size,
        'eval_batch_size': 20,
        'use_cuda': bool(opts.use_cuda),
        'early_stop': -1,
        'log_steps': 100,
        'opt_type': opts.opt_type
    }
    if opts.beta > 0:
        train_params['beta'] = opts.beta
    if opts.runner_type == 'coreset':
        runner = continual_runner.ContinualRunner(
            local_path=opts.local_path,
            model_params=model_params,
            transforms=None,
            train_params=train_params,
            selection_params=selection_params,
            use_cuda=bool(opts.use_cuda),
            task_dic=task_dic,
            buffer_size=opts.buffer_size,
            seed=opts.seed,
            replay_mode=opts.replay_mode,
            selection_transforms=eval_transforms if bool(opts.slt_wo_aug) else None,
            extra_data_mode=opts.extra_data.split(','),
            buffer_type=opts.buffer_type
        )
    else:
        raise ValueError('Invalid runner type')
    # continual training and update memory
    for i in range(generator.max_iter):
        if opts.runner_type == 'coreset':
            accs = runner.train_single_task(
                train_loader=train_loaders[i],
                eval_loaders=test_loaders,
                verbose=True,
                do_evaluation=True
            )
        else:
            raise ValueError('Invalid runner type')
        print('accuracies on testset after task', i, 'is:', accs, np.mean(accs))
        if opts.runner_type == 'coreset':
            runner.update_buffer(
                full_train_loader=train_loaders[i],
                sub_loader=train_loaders[i],
                next_loader=None
            )
        else:
            raise ValueError('Invalid runner type')
        runner.next_task(dump_buffer=True)


if __name__ == '__main__':
    """
    selection parameters are added in slt_config.py
    """
    parser = argparse.ArgumentParser('offline continual learning')
    parser.add_argument('--local_path', type=str)
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--setting', type=str, default='greedy')
    parser.add_argument('--buffer_size', type=int)
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--beta', type=float, default=0)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--mem_batch_size', type=int)
    parser.add_argument('--use_cuda', type=int)
    parser.add_argument('--opt_type', type=str)
    parser.add_argument('--slt_wo_aug', type=int, default=0)
    parser.add_argument('--holdout_set', type=str, default='full')
    parser.add_argument('--replay_mode', type=str, default='full')
    parser.add_argument('--use_bn', type=int, default=0)
    parser.add_argument('--limit_per_task', type=int, default=1000)
    parser.add_argument('--runner_type', type=str, default='coreset')
    parser.add_argument('--update_mode', type=str, default='random')
    parser.add_argument('--extra_data', type=str, default='')
    parser.add_argument('--slt_mse_factor', type=float, default=-1)
    parser.add_argument('--cur_train_steps', type=int, default=-1)
    parser.add_argument('--ref_train_epoch', type=int, default=-1)
    parser.add_argument('--selection_steps', type=int, default=-1)
    parser.add_argument('--ref_train_lr', type=float, default=-1)
    parser.add_argument('--cur_train_lr', type=float, default=-1)
    parser.add_argument('--aug_type', type=str, default='greedy')
    parser.add_argument('--buffer_type', type=str, default='coreset')
    parser.add_argument('--ref_sample_per_task', type=int, default=-1)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()

    utils.set_random_seed(seed=args.seed)
    print('script\t\t', 'rho_offline_continual_learning.py')
    print('local path\t\t', args.local_path)
    print('data path\t\t', args.data_path)
    print('dataset\t\t', args.dataset)
    print('setting\t\t', args.setting)
    print('buffer size\t\t', args.buffer_size)
    print('alpha\t\t', args.alpha)
    print('beta\t\t', args.beta)
    print('lr\t\t', args.lr)
    print('epochs\t\t', args.epochs)
    print('batch size\t\t', args.batch_size)
    print('mem batch size\t\t', args.mem_batch_size)
    print('opt type\t\t', args.opt_type)
    print('select without augmentation\t\t', args.slt_wo_aug)
    print('holdout set\t\t', args.holdout_set)
    print('replay mode\t\t', args.replay_mode)
    print('use bn\t\t', args.use_bn)
    print('limit per task\t\t', args.limit_per_task)
    print('runner type\t\t', args.runner_type)
    print('update mode\t\t', args.update_mode)
    print('extra data\t\t', args.extra_data)
    print('slt_mse_factor\t\t', args.slt_mse_factor)
    print('cur_train_steps\t\t', args.cur_train_steps)
    print('ref_train_epoch\t\t', args.ref_train_epoch)
    print('selection steps\t\t', args.selection_steps)
    print('ref train lr\t\t', args.ref_train_lr)
    print('cur train lr\t\t', args.cur_train_lr)
    print('aug type\t\t', args.aug_type)
    print('buffer type\t\t', args.buffer_type)
    print('ref sample per task\t\t', args.ref_sample_per_task)
    print('seed\t\t', args.seed)
    main(opts=args)

