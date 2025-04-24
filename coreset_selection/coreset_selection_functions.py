# -*-coding:utf8-*-

import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import random
import pickle
import copy

from dataset import single_task_dataset
from functions import loss_functions


def random_select(id2cnt, select_size):
    selected_id2prob = {}
    all_ids = list(id2cnt.keys())
    selected_ids = set(random.sample(all_ids, select_size))
    for d_id in selected_ids:
        selected_id2prob[d_id] = 1.0
    return selected_ids, selected_id2prob


def add_new_data(data_file, new_data):
    ori_data = []
    ori_ids = set()
    if os.path.exists(data_file):
        with open(data_file, 'rb') as fr:
            while True:
                try:
                    di = pickle.load(fr)
                    d_id = di[0]
                    ori_ids.add(int(d_id))
                    ori_data.append(di)
                except EOFError:
                    break
    all_data = ori_data
    for di in new_data:
        d_id = int(di[0])
        if d_id not in ori_ids:
            all_data.append(di)
    random.shuffle(all_data)
    with open(data_file, 'wb') as fw:
        for di in all_data:
            pickle.dump(di, fw)


def select_by_loss_diff(ref_loss_dic, rand_data, model, incremental_size, transforms, on_cuda, loss_params,
                        class_sizes=None):
    status = model.training
    model.eval()
    if on_cuda:
        model.cuda()
    loss_fn = loss_functions.CompliedLoss(
        ce_factor=loss_params['ce_factor'], mse_factor=loss_params['mse_factor'], reduction='none')
    loss_diffs = {}
    id2pos = {}
    id2logits = {}
    batch_ids = []
    batch_sps = []
    batch_labs = []
    batch_logits = []
    with torch.no_grad():
        for i, di in enumerate(rand_data):
            if len(di) == 4:
                d_id, sp, lab, logit = di
            else:
                d_id, sp, lab = di
                logit = None
            id2pos[d_id] = i
            if transforms is not None:
                aug_sp = torch.unsqueeze(transforms(sp), dim=0)
            else:
                aug_sp = torch.unsqueeze(sp, dim=0)
            batch_ids.append(d_id)
            batch_sps.append(aug_sp)
            batch_labs.append(int(lab))
            if logit is not None:
                batch_logits.append(
                    torch.unsqueeze(torch.tensor(logit, dtype=torch.float32), dim=0)
                )
            if i % 32 == 0 or i == len(rand_data) - 1:
                sps = torch.cat(batch_sps, dim=0)
                labs = torch.tensor(batch_labs, dtype=torch.long)
                if len(batch_logits) > 0:
                    lab_logits = torch.cat(batch_logits, dim=0)
                else:
                    lab_logits = None
                if on_cuda:
                    sps = sps.cuda()
                    labs = labs.cuda()
                    if lab_logits is not None:
                        lab_logits = lab_logits.cuda()
                loss = loss_fn(x=model(sps), y=labs, logits=lab_logits)
                loss = loss.clone().detach()
                if on_cuda:
                    loss = loss.cpu()
                loss = loss.numpy()
                if lab_logits is not None:
                    if on_cuda:
                        lab_logits = lab_logits.cpu()
                    lab_logits = lab_logits.clone().detach().numpy()
                for j in range(len(batch_labs)):
                    did = batch_ids[j]
                    loss_dif = float(loss[j] - ref_loss_dic[did])
                    loss_diffs[did] = loss_dif
                    if lab_logits is not None:
                        id2logits[did] = lab_logits[j, :]
                batch_ids.clear()
                batch_sps.clear()
                batch_labs.clear()
                batch_logits.clear()
                del lab_logits
    sorted_loss_diffs = sorted(loss_diffs.items(), key=lambda x: x[1], reverse=True)
    selected_data = []
    id2loss_dif = {}
    class_cnt = {}
    if class_sizes is not None:
        for ci in class_sizes.keys():
            class_cnt[ci] = 0
    for i in range(len(sorted_loss_diffs)):
        d_id = sorted_loss_diffs[i][0]
        pos = id2pos[d_id]
        di = rand_data[pos]
        if class_sizes is not None:
            lab = int(di[2])
            if class_cnt[lab] == class_sizes[lab]:
                continue
            else:
                class_cnt[lab] += 1
        new_di = copy.deepcopy(di)
        if loss_params['mse_factor'] > 0 and len(di) < 4:
            new_di.append(id2logits[d_id])
        selected_data.append(new_di)
        id2loss_dif[d_id] = sorted_loss_diffs[i][1]
        if len(selected_data) == incremental_size:
            break
    if on_cuda:
        model.cpu()
    model.train(status)
    return selected_data, id2loss_dif
