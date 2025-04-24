# -*-coding:utf8-*-

import torchvision
from torch.utils.data import DataLoader
import argparse
import os
import numpy as np
import pickle
import time

import utils
from coreset_selection import selection_agent
from dataset import single_task_dataset
from functions import train_methods


def get_datasets_cifar10(eval_transforms, data_path=''):
    if data_path == '':
        root_path = '../data'
    else:
        root_path = data_path
    to_tensor = torchvision.transforms.ToTensor()
    trainset = torchvision.datasets.CIFAR10(root=root_path, train=True, download=True, transform=to_tensor)
    trainset_wo_aug = torchvision.datasets.CIFAR10(
        root=root_path, train=True, download=False, transform=eval_transforms)
    testset = torchvision.datasets.CIFAR10(root=root_path, train=False, download=False, transform=eval_transforms)
    return trainset, trainset_wo_aug, testset


def get_datasets_mnist(eval_transforms, data_path=''):
    if data_path == '':
        root_path = '../data'
    else:
        root_path = data_path
    to_tensor = torchvision.transforms.ToTensor()
    train_dataset = torchvision.datasets.MNIST(root=root_path, train=True, transform=to_tensor, download=True)
    testset = torchvision.datasets.MNIST(root=root_path, train=False, transform=eval_transforms, download=True)
    return train_dataset, testset


def get_datasets_cifar100(eval_transforms, data_path=''):
    if data_path == '':
        root_path = '../data'
    else:
        root_path = data_path
    to_tensor = torchvision.transforms.ToTensor()
    trainset = torchvision.datasets.CIFAR100(root=root_path, train=True, download=True, transform=to_tensor)
    trainset_wo_aug = torchvision.datasets.CIFAR100(
        root=root_path, train=True, download=False, transform=eval_transforms)
    testset = torchvision.datasets.CIFAR100(root=root_path, train=False, download=False, transform=eval_transforms)
    return trainset, trainset_wo_aug, testset


def main(opts):
    if not os.path.exists(opts.local_path):
        os.makedirs(opts.local_path)
    # make-model-parameters, loss-parameters, transforms, ref-train-parameters
    # build full dataset, testset and subset
    if opts.dataset == 'cifar10':
        model_params = {
            'model_type': 'resnet',
            'num_class': 10,
            'num_blocks': [2, 2, 2, 2],
            'use_bn': bool(opts.use_bn)
        }
        train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        eval_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset, trainset_wo_aug, testset = get_datasets_cifar10(
            eval_transforms=eval_transforms, data_path=opts.data_path
        )
        test_loader = DataLoader(
            testset,
            batch_size=100,
            pin_memory=True,
            shuffle=False,
            drop_last=False
        )
    elif opts.dataset == 'cifar100':
        model_params = {
            'model_type': 'resnet',
            'num_class': 100,
            'num_blocks': [2, 2, 2, 2],
            'use_bn': bool(opts.use_bn)
        }
        train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        eval_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset, trainset_wo_aug, testset = get_datasets_cifar100(
            eval_transforms=eval_transforms, data_path=opts.data_path
        )
        test_loader = DataLoader(
            testset,
            batch_size=100,
            pin_memory=True,
            shuffle=False,
            drop_last=False
        )
    elif opts.dataset == 'mnist':
        model_params = {
            'model_type': 'cnn',
            'num_class': 10
        }
        train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
        eval_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
        trainset, testset = get_datasets_mnist(
            eval_transforms=eval_transforms, data_path=opts.data_path
        )
        test_loader = DataLoader(
            testset,
            batch_size=20,
            pin_memory=True,
            shuffle=False,
            drop_last=False
        )
    else:
        raise ValueError('Invalid dataset')
    loss_params = None
    ref_train_params = {
        'lr': opts.ref_lr,
        'epochs': opts.ref_epochs,
        'batch_size': 32,
        'eval_batch_size': 20,
        'use_cuda': bool(opts.use_cuda),
        'early_stop': -1,
        'log_steps': 100,
        'opt_type': 'sgd'
    }
    # build selection agent
    coreset_selector = selection_agent.RhoSelectionAgent(
        local_path=opts.local_path,
        transforms=train_transforms,
        init_size=opts.init_size,
        selection_steps=opts.selection_steps,
        cur_train_lr=opts.cur_train_lr,
        cur_train_steps=opts.cur_train_steps,
        use_cuda=bool(opts.use_cuda),
        eval_mode=opts.eval_mode,
        early_stop=opts.early_stop,
        eval_steps=opts.eval_steps,
        model_params=model_params,
        ref_train_params=ref_train_params,
        seed=opts.seed,
        ref_model=None,
        class_balance=bool(opts.class_balance),
        only_new_data=bool(opts.only_new_data),
        loss_params=None,
        save_checkpoint=bool(opts.save_checkpoint)
    )
    if len(opts.pool_path) > 0:
        pool_x = []
        pool_y = []
        to_tensor = torchvision.transforms.ToTensor()
        with open(opts.pool_path, 'rb') as fr:
            while True:
                try:
                    di = pickle.load(fr)
                    sp, lab = di
                    pool_x.append(to_tensor(sp).numpy())
                    pool_y.append(lab)
                except EOFError:
                    break
        pool_x = np.stack(pool_x, axis=0)
        pool_y = np.array(pool_y, dtype=np.int64)
        x = pool_x
        y = pool_y
    else:
        # get full data
        train_loader = DataLoader(trainset, batch_size=len(trainset), drop_last=False, shuffle=False)
        x, y = next(iter(train_loader))
        x = x.numpy()
        y = y.numpy()
        # get subset data
        if bool(opts.random_pool):
            all_inds = np.random.choice(np.arange(len(trainset)), opts.pool_size, replace=False)
        else:
            all_inds = np.arange(opts.pool_size)
        pool_x = x[all_inds]
        pool_y = y[all_inds]
    # train ref model
    if opts.ref_data == 'full':
        coreset_selector.train_ref_model(x=x, y=y)
    else:
        coreset_selector.train_ref_model(x=pool_x, y=pool_y)
    if bool(opts.save_checkpoint):
        save_file = os.path.join(opts.local_path, 'pool_data.pkl')
        with open(save_file, 'wb') as fw:
            pickle.dump([pool_x, pool_y], fw)
    # select coreset
    selected_data = coreset_selector.incremental_selection(
        x=pool_x,
        y=pool_y,
        select_size=opts.coreset_size
    )
    coreset_selector.clear_path()
    # train model on coreset
    test_train_file = os.path.join(opts.local_path, 'coreset.pkl')
    with open(test_train_file, 'wb') as fw:
        for di in selected_data:
            pickle.dump(di, fw)
    test_train_dataset = single_task_dataset.PILDataset(
        local_path=opts.local_path,
        data_path=test_train_file,
        transforms=train_transforms
    )
    test_train_dataset.set_produce_id(produce_id=True)
    test_train_loader = DataLoader(test_train_dataset, batch_size=opts.test_train_batch_size, drop_last=False)
    init_mdoel = utils.build_model(model_params=model_params)
    test_train_params = {
        'lr': opts.test_train_lr,
        'epochs': opts.test_train_epochs,
        'batch_size': 32,
        'eval_batch_size': 20,
        'use_cuda': bool(opts.use_cuda),
        'early_stop': -1,
        'log_steps': 100,
        'opt_type': opts.test_opt_type,
        'loss_params': {
            'ce_factor': 1.0,
            'mse_factor': 0.0
        }
    }
    trained_model = train_methods.train_model(
        local_path=opts.local_path,
        model=init_mdoel,
        train_loader=test_train_loader,
        eval_loader=test_loader,
        epochs=test_train_params['epochs'],
        train_params=test_train_params,
        verbose=True,
        save_ckpt=False,
        load_best=False,
        weight_decay=0
    )
    # evaluation
    acc = train_methods.eval_model(
        model=trained_model,
        eval_loader=test_loader,
        on_cuda=False,
        return_loss=False
    )
    print('accuracy on testset is:', acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('coreset selection')
    parser.add_argument('--local_path', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--use_cuda', type=int, default=1)
    parser.add_argument('--pool_size', type=int)
    parser.add_argument('--pool_path', type=str, default='')
    parser.add_argument('--random_pool', type=int, default=1)
    parser.add_argument('--init_size', type=int)
    parser.add_argument('--coreset_size', type=int)
    parser.add_argument('--ref_data', type=str, default='full')
    parser.add_argument('--ref_epochs', type=int)
    parser.add_argument('--ref_lr', type=float)
    parser.add_argument('--selection_steps', type=int)
    parser.add_argument('--cur_train_steps', type=int)
    parser.add_argument('--cur_train_lr', type=float)
    parser.add_argument('--eval_mode', type=str)
    parser.add_argument('--eval_steps', type=int)
    parser.add_argument('--early_stop', type=int)
    parser.add_argument('--only_new_data', type=int)
    parser.add_argument('--class_balance', type=int)
    parser.add_argument('--test_train_epochs', type=int)
    parser.add_argument('--test_train_lr', type=float)
    parser.add_argument('--test_opt_type', type=str, default='sgd')
    parser.add_argument('--test_train_batch_size', type=int, default=32)
    parser.add_argument('--use_bn', type=int, default=0)
    parser.add_argument('--save_checkpoint', type=int, default=0)
    parser.add_argument('--slt_aug', type=int, default=1)
    parser.add_argument('--rm_slt_step', type=int, default=0)
    parser.add_argument('--rm_interval', type=int, default=0)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()

    utils.set_random_seed(seed=args.seed)

    print('script\t\t', 'rho_data_summary.py')
    print('local path\t\t', args.local_path)
    print('dataset\t\t', args.dataset)
    print('data path\t\t', args.data_path)
    print('use cuda\t\t', args.use_cuda)
    print('pool size\t\t', args.pool_size)
    print('pool path\t\t', args.pool_path)
    print('random pool\t\t', args.random_pool)
    print('init size\t\t', args.init_size)
    print('coreset size\t\t', args.coreset_size)
    print('ref data mode\t\t', args.ref_data)
    print('ref epochs\t\t', args.ref_epochs)
    print('ref lr\t\t', args.ref_lr)
    print('selection steps\t\t', args.selection_steps)
    print('cur train steps\t\t', args.cur_train_steps)
    print('cur train lr\t\t', args.cur_train_lr)
    print('eval mode\t\t', args.eval_mode)
    print('eval steps\t\t', args.eval_steps)
    print('early stop\t\t', args.early_stop)
    print('only new data\t\t', args.only_new_data)
    print('class balance\t\t', args.class_balance)
    print('test train epochs\t\t', args.test_train_epochs)
    print('test train lr\t\t', args.test_train_lr)
    print('test opt type\t\t', args.test_opt_type)
    print('test train batch size\t\t', args.test_train_batch_size)
    print('use bn\t\t', args.use_bn)
    print('save checkpoint\t\t', args.save_checkpoint)
    print('select with augmentation\t\t', args.slt_aug)
    print('remove select steps\t\t', args.rm_slt_step)
    print('remove interval\t\t', args.rm_interval)
    print('seed\t\t', args.seed)

    t_start = time.perf_counter()
    print('start time:', t_start)
    main(opts=args)
    t_end = time.perf_counter()
    print('time cost is:', t_end - t_start)
    print('enc time:', t_end)
