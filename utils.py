# -*-coding:utf8-*-

import torch
from torch.utils.data import DataLoader
import random
import numpy as np
import torchvision
import copy
import os

import importlib
from dataset import single_task_dataset
from functions import loss_functions
from backbone import models
from backbone import resnets
from backbone import resnext
from backbone import tiny_vit


def set_random_seed(seed: int) -> None:
    """
    Sets the seeds at a certain value.
    :param seed: the value to be set
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    tv = torch.__version__
    if tv[:3] == '1.7' or tv[:3] == '1.8':
        torch.backends.cudnn.benchmark = False
        torch.set_deterministic(d=True)
    elif tv[:4] == '1.10' or tv[:4] == '1.13':
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        pass


def make_transforms(transform_list):
    """
    :param transform_list: each element is [transform type, dict-argument for transform]
    :return:
    """
    transforms = []
    for ti in transform_list:
        t_type = ti[0]
        trans_args = ti[1]
        if t_type == 'Resize':
            transforms.append(torchvision.transforms.Resize(**trans_args))
        elif t_type == 'RandomCrop':
            transforms.append(torchvision.transforms.RandomCrop(**trans_args))
        elif t_type == 'RandomHorizontalFlip':
            transforms.append(torchvision.transforms.RandomHorizontalFlip())
        elif t_type == 'Normalize':
            transforms.append(torchvision.transforms.Normalize(**trans_args))
        elif t_type == 'ToTensor':
            transforms.append(torchvision.transforms.ToTensor())
            print(t_type)
            raise ValueError('Not a valid transform type')
    transform = torchvision.transforms.Compose(transforms)
    return transform


def make_data_generator(dataset, train_path, test_path, eval_path=None):
    if dataset == 'seq_cifar':
        train_gen = torchvision.datasets.CIFAR10(
            root=train_path,
            train=True,
            download=False
        )
        test_gen = torchvision.datasets.CIFAR10(
            root=test_path,
            train=False,
            download=False
        )
        if eval_path is not None:
            eval_gen = torchvision.datasets.CIFAR10(
                root=eval_path,
                train=False,
                download=False
            )
        else:
            eval_gen = None
    elif dataset == 'seq_cifar_100':
        train_gen = torchvision.datasets.CIFAR100(
            root=train_path,
            train=True,
            download=False
        )
        test_gen = torchvision.datasets.CIFAR100(
            root=test_path,
            train=False,
            download=False
        )
        if eval_path is not None:
            eval_gen = torchvision.datasets.CIFAR100(
                root=eval_path,
                train=False,
                download=False
            )
        else:
            eval_gen = None
    elif dataset == 'perm_mnist' or dataset == 'rot_mnist' or dataset == 'seq_mnist':
        # print(train_path, eval_path, test_path)
        train_gen = torchvision.datasets.MNIST(
            root=train_path,
            train=True,
            download=False
        )
        test_gen = torchvision.datasets.MNIST(
            root=test_path,
            train=False,
            download=False
        )
        if eval_path is not None:
            eval_gen = torchvision.datasets.MNIST(
                root=eval_path,
                train=False,
                download=False
            )
        else:
            eval_gen = None
    else:
        raise ValueError('No such dataset in this implementation')
    return train_gen, test_gen, eval_gen


def build_model(model_params):
    if model_params['model_type'] == 'cnn':
        model = models.ConvNet(
            output_dim=model_params['num_class']
        )
    elif model_params['model_type'] == 'resnet':
        if 'setting' in model_params and model_params['setting'] == 'der':
            model = models.resnet18_der(nclasses=model_params['num_class'], nf=64)
        elif 'setting' in model_params and model_params['setting'] == 'large_model':
            model = resnets.resnet50(num_class=model_params['num_class'])
        else:
            if model_params['use_bn']:
                model = models.ResNet(
                    block=models.BasicBlockBN,
                    num_blocks=model_params['num_blocks'],
                    num_classes=model_params['num_class']
                )
            else:
                model = models.ResNet(
                    block=models.BasicBlock,
                    num_blocks=model_params['num_blocks'],
                    num_classes=model_params['num_class']
                )
    elif model_params['model_type'] == 'resnext':
        model = resnext.resnetxt50_32x4d(num_classes=model_params['num_class'])
    elif model_params['model_type'] == 'vit':
        model_kwargs = dict(
            img_size=64,  # for tiny-imagenet
            embed_dims=[64, 128, 256, 448],
            depths=[2, 2, 6, 2],
            num_heads=[2, 4, 8, 14],
            window_sizes=[7, 7, 14, 7],
            drop_path_rate=0.1,
            num_classes=model_params['num_class']
        )
        model = tiny_vit.TinyViT(**model_kwargs)
    elif model_params['model_type'] == 'mlp':
        model = models.FNNet(
            input_dim=model_params['input_dim'],
            interm_dim=model_params['interm_dim'],
            output_dim=model_params['num_class']
        )
    else:
        raise ValueError('Invalid model type')
    return model


def make_scheduler(scheduler_params, optimizer):
    if scheduler_params['type'] == 'multisteplr':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=scheduler_params['milestones'],
            gamma=scheduler_params['gamma']
        )
    else:
        print(scheduler_params['type'])
        raise ValueError('No such type of scheduler')
    return scheduler


def mask_classes(x, mask_ids):
    x[mask_ids] = -np.inf
    return x


def make_task_dic(total_class, class_split):
    task_dic = {}
    class_list = list(range(total_class))
    idx = 0
    for i, nc in enumerate(class_split):
        task_dic[i] = class_list[idx:idx + nc]
        idx += nc
    return task_dic


def compute_loss_dic(ref_model, data_loader, aug_iters, use_cuda, loss_params):
    ref_model.eval()
    loss_fn = loss_functions.CompliedLoss(
        ce_factor=loss_params['ce_factor'],
        mse_factor=loss_params['mse_factor'],
        reduction='none'
    )
    if use_cuda:
        ref_model.cuda()
    loss_dic = {}
    with torch.no_grad():
        for i in range(aug_iters):
            for data in data_loader:
                if len(data) == 4:
                    d_ids, sps, labs, logit = data
                else:
                    d_ids, sps, labs = data
                    logit = None
                if use_cuda:
                    sps = sps.cuda()
                    labs = labs.cuda()
                    if logit is not None:
                        logit = logit.cuda()
                loss = loss_fn(ref_model(sps), labs, logit)
                if use_cuda:
                    loss = loss.cpu()
                loss = loss.clone().detach().numpy()
                batch_size = sps.shape[0]
                for j in range(batch_size):
                    d_id = int(d_ids[j].numpy())
                    if d_id not in loss_dic:
                        loss_dic[d_id] = [loss[j]]
                    else:
                        loss_dic[d_id].append(loss[j])
    for d_id in loss_dic.keys():
        loss_dic[d_id] = float(np.mean(loss_dic[d_id]))
    if use_cuda:
        ref_model.cpu()
    return loss_dic


def compute_id2logit(data_loader, ref_model, aug_iters, use_cuda=True):
    status = ref_model.training
    ref_model.eval()
    if use_cuda:
        ref_model.cuda()
    id2logit = {}
    with torch.no_grad():
        for k in range(aug_iters):
            cur_id = 0
            for data in data_loader:
                if len(data) == 3:
                    _, sp, lab = data
                else:
                    sp, lab = data
                if use_cuda:
                    sp = sp.cuda()
                logits = ref_model(sp)
                if use_cuda:
                    logits = logits.cpu()
                logits = logits.clone().detach().numpy()
                for i in range(sp.shape[0]):
                    did = cur_id
                    logi = logits[i, :]
                    if did not in id2logit:
                        id2logit[did] = logi
                    else:
                        id2logit[did] = id2logit[did] + logi
                    cur_id += 1
    for d_id in id2logit.keys():
        id2logit[d_id] = (id2logit[d_id] / aug_iters)
    ref_model.train(status)
    if use_cuda:
        ref_model.cpu()
    return id2logit


def clear_dir(target_dir):
    for fi in os.listdir(target_dir):
        if fi.endswith('.pkl'):
            os.remove(os.path.join(target_dir, fi))


def count_parameters(model):
    num_params = 0
    num_train_params = 0
    for p in model.parameters():
        num_params += p.numel()
        if p.requires_grad:
            num_train_params += p.numel()
    return num_params, num_train_params
