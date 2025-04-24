# -*-coding:utf8-*-

import argparse
import pickle
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision
import os
import random
import numpy as np

import utils
import data_summary


class SimpleDataset(Dataset):
    def __init__(self, local_path, data, transforms, repeat=1):
        self.local_path = local_path
        self.data_file = os.path.join(local_path, 'train.pkl')
        self.data = []
        with open(self.data_file, 'wb') as fw:
            for i in range(repeat):
                for di in data:
                    pickle.dump(di, fw)
                    self.data.append(di)
        self.transforms = transforms

    def __getitem__(self, index):
        di = self.data[index]
        sp, lab = di
        if self.transforms is not None:
            aug_sp = self.transforms(sp)
        else:
            aug_sp = sp
        return aug_sp, lab

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        with open(self.data_file, 'rb') as fr:
            while True:
                try:
                    di = pickle.load(fr)
                    sp, lab = di
                    if self.transforms is not None:
                        aug_sp = self.transforms(sp)
                    else:
                        aug_sp = sp
                    yield aug_sp, lab
                except EOFError:
                    break

    def remove_datafile(self):
        if os.path.exists(self.data_file):
            os.remove(self.data_file)


def train_model(model, opt_type, epochs, lr, weight_decay, use_cuda, train_loader, eval_loader, last_n_acc=0):
    log_steps = 100
    loss_fn = torch.nn.CrossEntropyLoss()
    if use_cuda:
        model.cuda()
    if opt_type == 'adam':
        opt = torch.optim.Adam(lr=lr, params=model.parameters(), weight_decay=weight_decay)
    else:
        opt = torch.optim.SGD(lr=lr, params=model.parameters(), weight_decay=weight_decay)
    all_accs = []
    for i in range(epochs):
        # train model
        step = 0
        for data in train_loader:
            if len(data) == 2:
                sp, lab = data
            else:
                d_id, sp, lab = data
            if use_cuda:
                sp = sp.cuda()
                lab = lab.cuda()
            out = model(sp)
            loss = loss_fn(out, lab)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % log_steps == 0:
                print('loss at step:', step, 'is', loss.item())
            step += 1
        if i % 10 == 0 or i == epochs - 1:
            if eval_loader is not None:
                acc = eval_model(
                    model=model,
                    on_cuda=use_cuda,
                    eval_loader=eval_loader
                )
            else:
                acc = None
            print('accuracy in epoch', i, 'is:', acc)
        if last_n_acc > 0 and i >= (epochs - last_n_acc):
            if eval_loader is not None:
                acc = eval_model(
                    model=model,
                    on_cuda=use_cuda,
                    eval_loader=eval_loader
                )
                all_accs.append(acc)
        if hasattr(train_loader.dataset, 'shuffle_dataset'):
            train_loader.dataset.shuffle_dataset()
    if len(all_accs) > 0:
        print('average last accs is:', np.mean(all_accs), len(all_accs))
    if use_cuda:
        model = model.cpu()
    return model, np.mean(all_accs)


def eval_model(model, eval_loader, on_cuda):
    status = model.training
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(eval_loader):
            if on_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100. * correct / total
    model.train(status)
    return acc


def get_max_size(local_path):
    max_size = 0
    for fi in os.listdir(local_path):
        if 'selected_ids_' in fi:
            size = int(fi.split('.')[0].split('_')[-1])
            if size > max_size:
                max_size = size
    return max_size


def get_selected_data(pool_x, pool_y, inds):
    selected_data = []
    to_pil = torchvision.transforms.ToPILImage()
    for ind in inds:
        x = pool_x[ind, :]
        y = pool_y[ind]
        x_sp = to_pil(torch.tensor(x, dtype=torch.float32).clone().detach())
        y_lab = int(y)
        selected_data.append([x_sp, y_lab])
    return selected_data


def compute_average_loss(data_loader, model, use_cuda):
    status = model.training
    model.eval()
    if use_cuda:
        model.cuda()
    total_loss = 0
    total_cnt = 0
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    with torch.no_grad():
        for di in data_loader:
            sp, lab = di
            if use_cuda:
                sp = sp.cuda()
                lab = lab.cuda()
            loss = loss_fn(model(sp), lab)
            if use_cuda:
                loss = loss.cpu()
            loss = loss.clone().detach().numpy()
            total_loss += loss
            total_cnt += sp.shape[0]
    avg_loss = total_loss / total_cnt
    if use_cuda:
        model.cpu()
    model.train(status)
    return avg_loss


def main(opts):
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
        trainset, trainset_wo_aug, testset = data_summary.get_datasets_cifar10(
            eval_transforms=eval_transforms
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
        trainset, trainset_wo_aug, testset = data_summary.get_datasets_cifar100(
            eval_transforms=eval_transforms
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
        trainset, testset = data_summary.get_datasets_mnist(
            eval_transforms=eval_transforms
        )
        test_loader = DataLoader(
            testset,
            batch_size=100,
            pin_memory=True,
            shuffle=False,
            drop_last=False
        )
    else:
        raise ValueError('Invalid dataset')
    # load pool data
    pool_data_file = os.path.join(opts.local_path, 'pool_data.pkl')
    with open(pool_data_file, 'rb') as fr:
        data = pickle.load(fr)
    pool_x, pool_y = data
    if bool(opts.test_pool_loss):
        pool_data = get_selected_data(pool_x=pool_x, pool_y=pool_y, inds=np.arange(pool_x.shape[0]))
        pool_path = os.path.join(opts.local_path, 'pool_data')
        if not os.path.exists(pool_path):
            os.makedirs(pool_path)
        pool_dataset = SimpleDataset(
            local_path=pool_path,
            data=pool_data,
            transforms=eval_transforms,
            repeat=opts.repeat
        )
        pool_loader = DataLoader(pool_dataset, batch_size=100, shuffle=False, drop_last=False)
    else:
        pool_dataset = None
        pool_loader = None
    max_size = get_max_size(local_path=opts.local_path)
    size2perf = {}
    size2avg_loss = {}
    for i in range(opts.start_size, max_size + 1, opts.step_size):
        print('start evaluate size:', i)
        # load selected indices
        ind_file = os.path.join(opts.local_path, 'selected_ids_' + str(i) + '.pkl')
        with open(ind_file, 'rb') as fr:
            selected_inds = pickle.load(fr)
        selected_data = get_selected_data(pool_x=pool_x, pool_y=pool_y, inds=selected_inds)
        random.shuffle(selected_data)
        train_dataset = SimpleDataset(
            local_path=opts.local_path,
            data=selected_data,
            transforms=train_transforms,
            repeat=opts.repeat
        )
        train_loader = DataLoader(
            train_dataset, batch_size=opts.batch_size, drop_last=bool(opts.drop_last), shuffle=True)
        # initialize model and train model
        init_model = utils.build_model(model_params=model_params)
        trained_model, mean_acc = train_model(
            model=init_model,
            opt_type=opts.opt_type,
            epochs=opts.epochs,
            lr=opts.lr,
            weight_decay=opts.weight_decay,
            use_cuda=opts.use_cuda,
            train_loader=train_loader,
            eval_loader=test_loader,
            last_n_acc=opts.last_n_acc
        )
        # test_acc = eval_model(model=trained_model, eval_loader=test_loader, on_cuda=False)
        # print('accuracy on testset is:', test_acc)
        train_dataset.remove_datafile()
        size2perf[i] = mean_acc
        if pool_loader is not None:
            avg_loss = compute_average_loss(
                data_loader=pool_loader,
                model=trained_model,
                use_cuda=bool(opts.use_cuda)
            )
            print('average loss is:', avg_loss)
            size2avg_loss[i] = avg_loss
    if pool_dataset is not None:
        pool_dataset.remove_datafile()
    dump_file = os.path.join(opts.local_path, 'procedure_performance.pkl')
    with open(dump_file, 'wb') as fw:
        pickle.dump(size2perf, fw)
        pickle.dump(size2avg_loss, fw)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('train on coreset')
    parser.add_argument('--local_path', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--opt_type', type=str)
    parser.add_argument('--repeat', type=int)
    parser.add_argument('--drop_last', type=int)
    parser.add_argument('--use_cuda', type=int)
    parser.add_argument('--use_bn', type=int, default=0)
    parser.add_argument('--last_n_acc', type=int, default=0)
    parser.add_argument('--start_size', type=int, default=10)
    parser.add_argument('--step_size', type=int)
    parser.add_argument('--test_pool_loss', type=int, default=0)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()

    print('script\t\t', 'test_selected_data.py')
    print('local path\t\t', args.local_path)
    print('dataset\t\t', args.dataset)
    print('batch size\t\t', args.batch_size)
    print('epochs\t\t', args.epochs)
    print('lr\t\t', args.lr)
    print('weight decay\t\t', args.weight_decay)
    print('opt type\t\t', args.opt_type)
    print('repeat\t\t', args.repeat)
    print('drop_last\t\t', args.drop_last)
    print('use cuda\t\t', args.use_cuda)
    print('use_bn\t\t', args.use_bn)
    print('last n acc\t\t', args.last_n_acc)
    print('start size\t\t', args.start_size)
    print('step size\t\t', args.step_size)
    print('test pool loss\t\t', args.test_pool_loss)
    print('seed\t\t', args.seed)
    print('\n')
    utils.set_random_seed(seed=args.seed)
    main(opts=args)
