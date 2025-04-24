# -*-coding:utf8-*-

import argparse
import os
import numpy as np

import utils
from continual_learning import online_rho_runner
from continual_learning import online_er_runner
from dataset import stream_idataset


def main(opts):
    if not os.path.exists(opts.local_path):
        os.makedirs(opts.local_path)
    # make data loaders
    train_loaders, test_loaders, model_params, transforms = stream_idataset.get_data_loaders(
        dataset=opts.dataset,
        data_path=opts.data_path,
        batch_size=opts.batch_size,
        use_bn=bool(opts.use_bn)
    )
    if opts.setting != '':  # support for larger backbone experiment
        print('new model setting:', opts.setting)
        model_params['setting'] = opts.setting
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
    selection_params = {
        'slt_params': {
            'ce_factor': opts.ce_factor,
            'mse_factor': opts.mse_factor,
        },
        'cur_train_params': {
            'lr': opts.cur_train_lr,
            'steps': opts.cur_train_steps,
            'batch_size': 32,
            'eval_batch_size': 100,
            'use_cuda': bool(opts.use_cuda),
            'early_stop': -1,
            'log_steps': 100,
            'opt_type': 'sgd',
            'loss_params': {
                'ce_factor': opts.ce_factor,
                'mse_factor': opts.mse_factor,
            }
        },
        'selection_steps': opts.selection_steps
    }
    if 'beta' in train_params:
        runner = online_rho_runner.OnlineRhoRunner(
            local_path=opts.local_path,
            buffer_size=opts.buffer_size,
            model_params=model_params,
            use_cuda=bool(opts.use_cuda),
            transforms=transforms,
            selection_params=selection_params,
            train_params=train_params,
            update_mode=opts.update_mode,
            remove_mode=opts.remove_mode,
            scheduler_params=None,
            seed=opts.seed
        )
    else:
        scheduler_params = None
        if opts.dataset == 'splitcifar100':
            scheduler_params = {
                'type': 'multisteplr',
                'milestones': [35, 45],
                'gamma': 0.1
            }
        runner = online_er_runner.OnlineERRunner(
            local_path=opts.local_path,
            buffer_size=opts.buffer_size,
            model_params=model_params,
            use_cuda=bool(opts.use_cuda),
            transforms=transforms,
            selection_params=selection_params,
            train_params=train_params,
            update_mode=opts.update_mode,
            remove_mode=opts.remove_mode,
            scheduler_params=scheduler_params,
            seed=opts.seed
        )
    for i in range(len(train_loaders)):
        accs = runner.train_single_task(
            train_loader=train_loaders[i],
            eval_loaders=test_loaders,
            verbose=True,
            do_evaluation=True
        )
        print('accuracies on testset after task', i, 'is:', accs, np.mean(accs))
        runner.next_task(dump_buffer=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('rho stream continual learning')
    parser.add_argument('--local_path', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--buffer_size', type=int)
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--beta', type=float, default=0)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--mem_batch_size', type=int)
    parser.add_argument('--use_cuda', type=int)
    parser.add_argument('--opt_type', type=str)
    parser.add_argument('--use_bn', type=int)
    parser.add_argument('--ce_factor', type=float)
    parser.add_argument('--mse_factor', type=float)
    parser.add_argument('--update_mode', type=str, default='rho_loss')
    parser.add_argument('--remove_mode', type=str, default='random')
    parser.add_argument('--cur_train_lr', type=float)
    parser.add_argument('--cur_train_steps', type=int)
    parser.add_argument('--selection_steps', type=int)
    parser.add_argument('--setting', default='')
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()

    utils.set_random_seed(seed=args.seed)
    print('script\t\t', 'stream_continual_learning.py')
    print('local path\t\t', args.local_path)
    print('dataset\t\t', args.dataset)
    print('data path\t\t', args.data_path)
    print('buffer size\t\t', args.buffer_size)
    print('alpha\t\t', args.alpha)
    print('beta\t\t', args.beta)
    print('lr\t\t', args.lr)
    print('epochs\t\t', args.epochs)
    print('batch size\t\t', args.batch_size)
    print('mem batch size\t\t', args.mem_batch_size)
    print('use cuda\t\t', args.use_cuda)
    print('opt type\t\t', args.opt_type)
    print('use bn\t\t', args.use_bn)
    print('ce factor\t\t', args.ce_factor)
    print('mse factor\t\t', args.mse_factor)
    print('update mode\t\t', args.update_mode)
    print('remove mode\t\t', args.remove_mode)
    print('cur train lr\t\t', args.cur_train_lr)
    print('cur train steps\t\t', args.cur_train_steps)
    print('selection steps\t\t', args.selection_steps)
    print('setting\t\t', args.setting)
    print('seed\t\t', args.seed)
    print('\n')
    main(opts=args)
