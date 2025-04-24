# -*-coding:utf8-*-

import torch
from torch.utils.data import DataLoader
import torchvision
import pickle
import os
import random
import copy

import utils
from dataset import single_task_dataset
from coreset_selection import coreset_selection_functions
from coreset_selection import train_methods_for_selection
from functions import train_methods


class RhoSelectionAgent(object):
    def __init__(self, local_path, transforms, init_size, selection_steps, cur_train_lr, cur_train_steps, use_cuda,
                 eval_mode, early_stop, eval_steps, model_params, ref_train_params, seed, ref_model=None,
                 class_balance=True, only_new_data=True, loss_params=None, save_checkpoint=False):
        # all related setting
        self.local_path = local_path
        if not os.path.exists(self.local_path):
            os.makedirs(self.local_path)
        self.transforms = transforms
        self.init_size = init_size
        self.selection_steps = selection_steps
        self.cur_train_lr = cur_train_lr
        self.cur_train_steps = cur_train_steps
        self.eval_mode = eval_mode
        self.early_stop = early_stop
        self.eval_steps = eval_steps
        self.class_balance = class_balance
        self.only_new_data = only_new_data
        self.loss_params = loss_params
        self.ref_model = ref_model
        self.ref_train_params = ref_train_params
        self.model_params = model_params
        self.seed = seed
        self.save_checkpoint = save_checkpoint
        # make train_params
        if loss_params is None:
            loss_params = {
                'ce_factor': 1.0,
                'mse_factor': 0.0
            }
        self.train_params = {
            'lr': self.cur_train_lr,
            'steps': self.cur_train_steps,
            'batch_size': 32,
            'eval_batch_size': 20,
            'use_cuda': use_cuda,
            'early_stop': self.early_stop,
            'log_steps': 100,
            'opt_type': 'sgd',
            'loss_params': loss_params
        }
        self.cur_train_file = os.path.join(self.local_path, 'cur_train.pkl')
        self.to_pil = torchvision.transforms.ToPILImage()

    def make_data_loader(self, x, y, fname, batch_size, id_list=None, id2logit=None, extra_data=None):
        """
        make train-dataloader from numpy array inputs
        :param x: input numpy array
        :param y: target numpy array
        :param fname:
        :param batch_size:
        :param id_list:
        :param id2logit:
        :param extra_data:
        :return:
        """
        data_size = x.shape[0]
        data_file = os.path.join(self.local_path, fname)
        with open(data_file, 'wb') as fw:
            for i in range(data_size):
                if self.transforms is not None:
                    sp = self.to_pil(torch.tensor(x[i], dtype=torch.float32).clone().detach())
                else:
                    sp = torch.tensor(x[i], dtype=torch.float32).clone().detach()
                if id_list is not None:
                    data = [id_list[i], sp, int(y[i])]
                    if id2logit is not None:
                        data.append(id2logit[id_list[i]])
                else:
                    data = [i, sp, int(y[i])]
                    if id2logit is not None:
                        data.append(id2logit[i])
                pickle.dump(data, fw)
            if extra_data is not None:
                for di in extra_data:
                    if len(di) == 4 and id2logit is None:
                        pickle.dump(di[:3], fw)
                    else:
                        pickle.dump(di, fw)
        dataset = single_task_dataset.PILDataset(
            local_path=self.local_path,
            data_path=data_file,
            transforms=self.transforms
        )
        dataset.set_produce_id(produce_id=True)
        dataset.shuffle_dataset()
        data_loader = DataLoader(dataset, batch_size=batch_size, drop_last=False)
        return data_loader, data_file

    def train_ref_model(self, x, y, verbose=True, id2logit=None, ideal_logit=False, extra_data=None, log_file=None):
        print('=== train holdout model ===')
        train_loader, data_file = self.make_data_loader(
            x=x,
            y=y,
            fname='ref_train.pkl',
            batch_size=self.ref_train_params['batch_size'],
            id2logit=id2logit,
            extra_data=extra_data
        )
        init_model = utils.build_model(model_params=self.model_params)
        if ideal_logit and 'loss_params' in self.ref_train_params:
            temp_train_params = copy.deepcopy(self.ref_train_params)
            temp_train_params['loss_params'] = {
                'ce_factor': 1.0,
                'mse_factor': 0.0
            }
        else:
            temp_train_params = self.ref_train_params
        trained_model = train_methods.train_model(
            local_path=self.local_path,
            model=init_model,
            train_loader=train_loader,
            eval_loader=None,
            epochs=self.ref_train_params['epochs'],
            train_params=temp_train_params,
            verbose=verbose,
            save_ckpt=False,
            load_best=False,
            weight_decay=0,
            log_file=log_file
        )
        os.remove(data_file)
        self.ref_model = trained_model

    def incremental_selection(self, x, y, select_size, id_list=None, loss_dic=None, loss_dic_dump_file=None,
                              verbose=True, class_pool=None, id2logit=None, ideal_logit=False, extra_data=None):
        if select_size >= x.shape[0]:
            print('Warning: select size greater than data size', select_size, x.shape[0])
        # the id of each sample is assigned according to order.
        if loss_dic is None:
            # train reference model
            if self.ref_model is None:
                self.train_ref_model(x=x, y=y, id2logit=id2logit)
            # compute loss dict
            temp_loader, temp_file = self.make_data_loader(
                x=x,
                y=y,
                fname='temp_data.pkl',
                batch_size=20,
                id_list=id_list,
                id2logit=id2logit
            )
            if ideal_logit:
                ref_loss_params = {
                    'ce_factor': 1.0,
                    'mse_factor': 0.0
                }
            else:
                ref_loss_params = self.train_params['loss_params']
            ref_loss_dic = utils.compute_loss_dic(
                ref_model=self.ref_model,
                data_loader=temp_loader,
                aug_iters=1,
                use_cuda=self.train_params['use_cuda'],
                loss_params=ref_loss_params
            )
            if loss_dic_dump_file is not None:
                with open(loss_dic_dump_file, 'wb') as fw:
                    pickle.dump(ref_loss_dic, fw)
            os.remove(temp_file)
        else:
            ref_loss_dic = loss_dic
        # init model and selection
        all_selected_ids = set()
        incremental_size = max(int(select_size / self.selection_steps), 1)
        init_model = utils.build_model(model_params=self.model_params)
        all_class_ids = get_class_dic(y=y)
        class_ids = {}
        if class_pool is None:
            for i in range(self.model_params['num_class']):
                class_ids[i] = set()
        else:
            for i in class_pool:
                class_ids[i] = set()
        cur_train_dataset = single_task_dataset.RandomDataset(
            seed=self.seed,
            data_path=self.cur_train_file,
            transforms=self.transforms,  # the transforms can be altered here
            extra_data=extra_data
        )
        train_loader = DataLoader(cur_train_dataset, batch_size=self.train_params['batch_size'], drop_last=False)
        if self.eval_mode in ['acc', 'avg_loss', 'loss_var']:
            full_train_loader, full_data_file = self.make_data_loader(
                x=x,
                y=y,
                batch_size=self.train_params['batch_size'],
                fname='full_data.pkl',
                id_list=id_list,
                id2logit=id2logit
            )
        else:
            full_train_loader = None
            full_data_file = ''
        if id_list is None:
            full_ids = list(range(x.shape[0]))
        else:
            full_ids = id_list
        # make initial set
        if self.init_size > 0:
            if bool(self.class_balance):
                class_size = []
                base_size = int(self.init_size // self.model_params['num_class'])
                for i in range(self.model_params['num_class']):
                    class_size.append(base_size)
                res = self.init_size - self.model_params['num_class'] * base_size
                for i in range(self.model_params['num_class']):
                    if res == 0:
                        break
                    class_size[i] = class_size[i] + 1
                    res -= 1
                init_ids = set()
                for i in range(self.model_params['num_class']):
                    cids = all_class_ids[i]
                    init_cids = random.sample(list(cids), class_size[i])
                    for d_id in init_cids:
                        init_ids.add(d_id)
            else:
                init_ids = random.sample(full_ids, self.init_size)
                init_ids = set(init_ids)
            init_data = get_subset_by_id(
                x=x, y=y, ids=init_ids,
                transforms=self.to_pil if self.transforms is not None else None,
                id_list=id_list)
            for di in init_data:
                d_id = int(di[0])
                lab = int(di[2])
                class_ids[lab].add(d_id)
            with open(self.cur_train_file, 'wb') as fw:
                for di in init_data:
                    pickle.dump(di, fw)
            for d_id in init_ids:
                all_selected_ids.add(d_id)
            if self.save_checkpoint:
                self.dump_selected_ids(selected_ids=all_selected_ids)
            result = train_methods_for_selection.train_model(
                local_path=self.local_path,
                model=init_model,
                train_loader=train_loader,
                train_params=self.train_params,
                eval_loader=full_train_loader,
                eval_mode=self.eval_mode,
                verbose=verbose,
                load_best=False,
                eval_steps=self.eval_steps
            )
            init_model = result
        while len(all_selected_ids) < select_size:
            id_pool = set()
            for d_id in full_ids:
                if bool(self.only_new_data):
                    if d_id not in all_selected_ids:
                        id_pool.add(d_id)
                else:
                    id_pool.add(d_id)
            if bool(self.class_balance):
                class_sizes = make_class_sizes(
                    class_ids=class_ids,
                    incremental_size=min(incremental_size, select_size - len(all_selected_ids))
                )
            else:
                class_sizes = None
            rand_data = get_subset_by_id(
                x=x,
                y=y,
                ids=id_pool,
                transforms=self.to_pil if self.transforms is not None else None,
                id_list=id_list,
                id2logit=id2logit
            )
            selected_data, _ = coreset_selection_functions.select_by_loss_diff(
                ref_loss_dic=ref_loss_dic,
                rand_data=rand_data,
                model=init_model,
                incremental_size=min(incremental_size, select_size - len(all_selected_ids)),
                transforms=self.transforms,
                on_cuda=self.train_params['use_cuda'],
                loss_params=self.train_params['loss_params'],
                class_sizes=class_sizes
            )
            flg_add = False
            for di in selected_data:
                d_id = int(di[0])
                lab = int(di[2])
                if d_id not in all_selected_ids:
                    all_selected_ids.add(d_id)
                    flg_add = True
                class_ids[lab].add(d_id)
            if flg_add:
                coreset_selection_functions.add_new_data(data_file=self.cur_train_file, new_data=selected_data)
                cur_train_dataset.load_data()
                remove_ids = set()
                for di in selected_data:
                    d_id = int(di[0])
                    remove_ids.add(d_id)
                if full_train_loader is not None:
                    full_train_loader.dataset.remove_data_by_id(ids=remove_ids)
                init_model = utils.build_model(model_params=self.model_params)
                if self.save_checkpoint:
                    self.dump_selected_ids(selected_ids=all_selected_ids)
            result = train_methods_for_selection.train_model(
                local_path=self.local_path,
                model=init_model,
                train_loader=train_loader,
                train_params=self.train_params,
                eval_loader=full_train_loader,
                eval_mode=self.eval_mode,
                verbose=verbose,
                load_best=False,
                eval_steps=self.eval_steps
            )
            trained_model = result
            init_model = trained_model
            print('finish selecting samples:', len(all_selected_ids))
        if len(full_data_file) > 0:
            os.remove(full_data_file)
        # get selected data
        selected_data = []
        with open(self.cur_train_file, 'rb') as fr:
            while True:
                try:
                    di = pickle.load(fr)
                    selected_data.append(di)
                except EOFError:
                    break
        return selected_data

    def clear_path(self):
        if os.path.exists(self.cur_train_file):
            os.remove(self.cur_train_file)

    def reset_ref_model(self):
        self.ref_model = None

    def dump_selected_ids(self, selected_ids):
        num_sps = len(selected_ids)
        dump_file = os.path.join(self.local_path, 'selected_ids_' + str(num_sps) + '.pkl')
        with open(dump_file, 'wb') as fw:
            pickle.dump(selected_ids, fw)


def get_class_dic(y):
    class_dic = {}
    for i in range(y.shape[0]):
        lab = int(y[i])
        if lab not in class_dic:
            class_dic[lab] = [i]
        else:
            class_dic[lab].append(i)
    return class_dic


def get_subset_by_id(x, y, ids, transforms=None, id_list=None, id2logit=None):
    selected_data = []
    if id_list is None:
        d_pos = ids
    else:
        d_pos = []
        id_pool = set(ids)
        for i, d_id in enumerate(id_list):
            if d_id in id_pool:
                d_pos.append(i)
    for pi in d_pos:
        if transforms is None:
            sp = torch.tensor(x[pi], dtype=torch.float32).clone().detach()
        else:
            sp = transforms(torch.tensor(x[pi], dtype=torch.float32).clone().detach())
        if id_list is None:
            d_id = pi
        else:
            d_id = id_list[pi]
        data = [d_id, sp, int(y[pi])]
        if id2logit is not None:
            data.append(id2logit[d_id])
        selected_data.append(data)
    return selected_data


def make_class_sizes(class_ids, incremental_size):
    class_cnts = {}
    max_cnt = -1
    for ci in class_ids.keys():
        class_cnts[ci] = len(class_ids[ci])
        if len(class_ids[ci]) > max_cnt:
            max_cnt = len(class_ids[ci])
    sorted_cnt = sorted(class_cnts.items(), key=lambda x: x[1])
    class_sizes = {}
    for ci in class_ids.keys():
        class_sizes[ci] = 0
    rest_size = incremental_size
    for i in range(len(sorted_cnt)):
        ci, cnt = sorted_cnt[i]
        to_select = max_cnt - cnt
        to_select = min(to_select, rest_size)
        class_sizes[ci] = to_select
        rest_size -= to_select
        if rest_size == 0:
            break
    if rest_size > 0:
        base_size = int(rest_size // len(class_ids))
        for ci in class_sizes.keys():
            class_sizes[ci] = class_sizes[ci] + base_size
        rest_size = rest_size - base_size * len(class_ids)
        for ci in class_sizes.keys():
            class_sizes[ci] += 1
            rest_size -= 1
            if rest_size == 0:
                break
    return class_sizes
