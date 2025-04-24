# -*-coding:utf8-*-

import copy
import torch
import pickle
import os
import numpy as np
import torchvision
import random

from coreset_selection import selection_agent
import utils


class CoresetBuffer(object):
    def __init__(self, local_path, model_params, transforms, selection_params, buffer_size, use_cuda,
                 task_dic, seed, selection_transforms=None, extra_data_mode=None):
        self.local_path = local_path
        if not os.path.exists(self.local_path):
            os.makedirs(self.local_path)
        self.model_params = model_params
        self.transforms = transforms
        self.selection_params = selection_params
        self.use_cuda = use_cuda
        self.seed = seed
        self.buffer_size = buffer_size
        self.task_dic = task_dic
        self.selection_transforms = selection_transforms
        self.extra_data_mode = extra_data_mode
        # build coreset selector
        self.coreset_selector = selection_agent.RhoSelectionAgent(
            local_path=self.local_path,
            transforms=self.transforms if self.selection_transforms is None else self.selection_transforms,
            init_size=0,
            selection_steps=self.selection_params['selection_steps'],
            cur_train_lr=self.selection_params['cur_train_lr'],
            cur_train_steps=self.selection_params['cur_train_steps'],
            use_cuda=self.use_cuda,
            eval_mode='none',
            early_stop=-1,
            eval_steps=100,
            model_params=self.model_params,
            ref_train_params=self.selection_params['ref_train_params'],
            seed=self.seed,
            ref_model=None,
            class_balance=self.selection_params['class_balance'],
            only_new_data=True,
            loss_params=None if 'loss_params' not in self.selection_params else self.selection_params['loss_params']
        )
        self.data = []
        self.id2task = {}
        self.to_tensor = torchvision.transforms.ToTensor()
        self.to_pil = torchvision.transforms.ToPILImage()
        self.id_bias = 0

    def update_buffer(self, task_cnts, task_id, cur_x, cur_y, full_cur_x, full_cur_y, cur_id2logit=None,
                      next_x=None, next_y=None):
        # distribute buffer size to each task
        task_sizes = []
        for i in range(task_id + 1):
            task_sizes.append(int(task_cnts[i] / sum(task_cnts) * self.buffer_size))
        rest_size = self.buffer_size - sum(task_sizes)
        for j in range(rest_size):
            task_sizes[j] += 1
        new_id2task = {}
        new_data = []
        for i in range(task_id + 1):
            new_data.append([])
        pre_select_size = 0
        # make id list
        cur_id_list = []
        for i in range(cur_x.shape[0]):
            cur_id_list.append(i + self.id_bias)
        if cur_id2logit is not None:  # support for adding knowledge distillation
            new_id2logit = {}
            for d_id in cur_id2logit.keys():
                new_did = d_id + self.id_bias
                new_id2logit[new_did] = cur_id2logit[d_id]
        else:
            new_id2logit = None
        # re-select previous tasks
        for i in range(task_id):
            print('\tselect coreset for task', i, task_sizes[i])
            # get data
            id_pool = []
            prv_x = []
            prv_y = []
            id2logit = {}
            for di in self.data[i]:
                if len(di) == 3:
                    d_id, sp, lab = di
                elif len(di) == 4:
                    d_id, sp, lab, logit = di
                    id2logit[d_id] = logit
                else:
                    raise ValueError('Invalid data length')
                if isinstance(sp, torch.Tensor):
                    prv_x.append(sp.numpy())
                else:
                    prv_x.append(self.to_tensor(sp).numpy())
                prv_y.append(lab)
                id_pool.append(d_id)
            prv_x = np.stack(prv_x, axis=0)
            prv_y = np.array(prv_y)
            if len(id2logit) == 0:
                id2logit = None
            # load loss dic
            loss_dic_file = os.path.join(self.local_path, 'ref_loss_dic' + str(i) + '.pkl')
            with open(loss_dic_file, 'rb') as fr:
                ref_loss_dic = pickle.load(fr)
            # add support for extra data
            extra_data = self.make_extra_data(
                cur_x=cur_x,
                cur_y=cur_y,
                cur_id_list=cur_id_list,
                task_id=task_id,
                c_tid=i,
                task_sizes=task_sizes,
                next_x=next_x,
                next_y=next_y
            )
            # coreset selection
            selected_data = self.coreset_selector.incremental_selection(
                x=prv_x,
                y=prv_y,
                select_size=task_sizes[i],
                loss_dic=ref_loss_dic,
                verbose=False,
                class_pool=self.task_dic[i],
                id_list=id_pool,
                id2logit=id2logit,
                extra_data=extra_data
            )
            self.coreset_selector.clear_path()
            # update data and id2task
            for si in selected_data:
                new_data[i].append(si)
                d_id = int(si[0])
                new_id2task[d_id] = i
                pre_select_size += 1
            id_pool.clear()
        # select current task data
        cur_select_size = self.buffer_size - pre_select_size
        # train current ref model
        print('\ttrain ref model for task', task_id, cur_select_size)
        if 'ref_sample_per_task' in self.selection_params['ref_train_params'] and \
                self.selection_params['ref_train_params']['ref_sample_per_task'] > 0:
            extra_data = self.make_extra_ref_samples(
                sample_per_task=self.selection_params['ref_train_params']['ref_sample_per_task'])
        else:
            extra_data = None
        self.coreset_selector.train_ref_model(
            x=full_cur_x,
            y=full_cur_y,
            verbose=False,
            extra_data=extra_data,
            log_file=os.path.join(self.local_path, 'holdout_model_loss' + str(task_id) + '.pkl')
        )
        loss_dic_dump_file = os.path.join(self.local_path, 'ref_loss_dic' + str(task_id) + '.pkl')
        # coreset selection
        print('\tselect coreset for task', task_id)
        extra_data = self.make_extra_data(
            cur_x=cur_x,
            cur_y=cur_y,
            cur_id_list=cur_id_list,
            task_id=task_id,
            c_tid=task_id,
            task_sizes=task_sizes,
            next_x=next_x,
            next_y=next_y
        )
        cur_selected_data = self.coreset_selector.incremental_selection(
            x=cur_x,
            y=cur_y,
            select_size=cur_select_size,
            loss_dic=None,
            loss_dic_dump_file=loss_dic_dump_file,
            verbose=False,
            class_pool=self.task_dic[task_id],
            id_list=cur_id_list,
            id2logit=new_id2logit,
            extra_data=extra_data
        )
        self.coreset_selector.clear_path()
        self.id_bias += cur_x.shape[0]
        self.coreset_selector.reset_ref_model()
        # update data and id2task
        for si in cur_selected_data:
            new_data[task_id].append(si)
            d_id = int(si[0])
            new_id2task[d_id] = task_id
            pre_select_size += 1
        # update data
        self.data = new_data
        self.id2task = new_id2task

    def make_extra_data(self, cur_x, cur_y, cur_id_list, task_id, c_tid, task_sizes, next_x=None, next_y=None):
        extra_data = []
        if self.extra_data_mode is not None and 'other_task' in self.extra_data_mode:
            for j in range(task_id):  # for previous data
                if j == c_tid:
                    continue
                for di in self.data[j]:
                    extra_data.append(di)
            # for current data
            if c_tid < task_id:
                rand_cur_ids = random.sample(cur_id_list, len(self.data[c_tid]))
                rand_cur_data = selection_agent.get_subset_by_id(
                    x=cur_x,
                    y=cur_y,
                    ids=rand_cur_ids,
                    transforms=self.to_pil if self.transforms is not None else None,
                    id_list=cur_id_list,
                    id2logit=None
                )
                extra_data = extra_data + rand_cur_data
        if self.extra_data_mode is not None and 'next_task' in self.extra_data_mode:
            if next_x is None or next_y is None:
                extra_data = []
            else:
                next_id_list = []
                for j in range(next_x.shape[0]):
                    next_id_list.append(j + self.id_bias + cur_x.shape[0])
                rand_next_ids = random.sample(next_id_list, task_sizes[c_tid])
                rand_next_data = selection_agent.get_subset_by_id(
                    x=next_x,
                    y=next_y,
                    ids=rand_next_ids,
                    transforms=self.to_pil if self.transforms is not None else None,
                    id_list=next_id_list,
                    id2logit=None
                )
                extra_data = extra_data + rand_next_data
        if len(extra_data) == 0:
            extra_data = None
        return extra_data

    def make_extra_ref_samples(self, sample_per_task):
        extra_data = []
        for i in range(len(self.data)):
            if len(self.data[i]) >= sample_per_task:
                sub_data = random.sample(self.data[i], sample_per_task)
                extra_data = extra_data + copy.deepcopy(sub_data)
            else:
                resample_time = int(sample_per_task // len(self.data[i]))
                res = sample_per_task - resample_time * len(self.data[i])
                for j in range(resample_time):
                    extra_data = extra_data + copy.deepcopy(self.data[i])
                extra_data = extra_data + copy.deepcopy(self.data[i][:res])
        random.shuffle(extra_data)
        return extra_data

    def get_data(self):
        for i in range(len(self.data)):
            sps = []
            labs = []
            logits = []
            for di in self.data[i]:
                if len(di) == 3:
                    d_id, sp, lab = di
                elif len(di) == 4:
                    d_id, sp, lab, logit = di
                    logits.append(torch.tensor(logit, dtype=torch.float32))
                else:
                    raise ValueError('Invalid data length')
                if self.transforms is not None:
                    aug_sp = self.transforms(sp)
                else:
                    aug_sp = sp
                sps.append(aug_sp)
                labs.append(lab)
            out_data = [torch.stack(sps, dim=0), torch.tensor(labs, dtype=torch.long)]
            if len(logits) > 0:
                out_data.append(torch.stack(logits, dim=0))
            yield out_data

    def get_sub_data(self, size):
        all_data = []
        for i in range(len(self.data)):
            all_data = all_data + self.data[i]
        inds = random.sample(list(range(len(all_data))), size)
        selected_sps = []
        selected_labs = []
        selected_logits = []
        for idx in inds:
            di = all_data[idx]
            if len(di) == 3:
                d_id, sp, lab = di
            elif len(di) == 4:
                d_id, sp, lab, logit = di
                selected_logits.append(torch.tensor(logit, dtype=torch.float32))
            else:
                raise ValueError('Invalid data length')
            if self.transforms is not None:
                aug_sp = self.transforms(sp)
            else:
                aug_sp = sp
            selected_sps.append(aug_sp)
            selected_labs.append(lab)
        selected_sps = torch.stack(selected_sps, dim=0)
        selected_labs = torch.tensor(selected_labs, dtype=torch.long)
        if len(selected_logits) > 0:
            selected_logits = torch.stack(selected_logits, dim=0)
            return selected_sps, selected_labs, selected_logits
        else:
            return selected_sps, selected_labs

    def shuffle_data(self):
        random.shuffle(self.data)

    def is_empty(self):
        if len(self.data) == 0:
            return True
        else:
            return False

    def dump_data(self, task_id):
        dump_file = os.path.join(self.local_path, 'buffer_data' + str(task_id) + '.pkl')
        with open(dump_file, 'wb') as fw:
            pickle.dump(self.data, fw)
            pickle.dump(self.id2task, fw)
            pickle.dump(self.id_bias, fw)


class UniformBuffer(object):
    def __init__(self, local_path, transforms, buffer_size, use_cuda, seed):
        self.local_path = local_path
        if not os.path.exists(self.local_path):
            os.makedirs(self.local_path)
        self.transforms = transforms
        self.buffer_size = buffer_size
        self.use_cuda = use_cuda
        self.seed = seed
        self.data = []

    def update_buffer(self, task_cnts, task_id, cur_x, cur_y):
        # distribute buffer size to each task
        task_sizes = []
        for i in range(task_id + 1):
            task_sizes.append(int(task_cnts[i] / sum(task_cnts) * self.buffer_size))
        rest_size = self.buffer_size - sum(task_sizes)
        for j in range(rest_size):
            task_sizes[j] += 1
        # resize previous task data
        for i in range(task_id):
            new_data = random.sample(self.data[i], task_sizes[i])
            self.data[i] = copy.deepcopy(new_data)
        # select current task data
        selected_ids = random.sample(list(range(cur_x.shape[0])), task_sizes[task_id])
        to_pil = torchvision.transforms.ToPILImage()
        cur_data = []
        for idx in selected_ids:
            if self.transforms is not None:
                sp = to_pil(torch.tensor(cur_x[idx], dtype=torch.float32).clone().detach())
            else:
                sp = torch.tensor(cur_x[idx], dtype=torch.float32).clone().detach()
            lab = int(cur_y[idx])
            cur_data.append([sp, lab])
        self.data.append(cur_data)

    def get_data(self):
        for i in range(len(self.data)):
            sps = []
            labs = []
            for di in self.data[i]:
                sp, lab = di
                if self.transforms is not None:
                    aug_sp = self.transforms(sp)
                else:
                    aug_sp = sp
                sps.append(aug_sp)
                labs.append(lab)
            out_data = [torch.stack(sps, dim=0), torch.tensor(labs, dtype=torch.long)]
            yield out_data

    def get_sub_data(self, size):
        all_data = []
        for i in range(len(self.data)):
            all_data = all_data + self.data[i]
        inds = random.sample(list(range(len(all_data))), size)
        selected_sps = []
        selected_labs = []
        for idx in inds:
            di = all_data[idx]
            sp, lab = di
            if self.transforms is not None:
                aug_sp = self.transforms(sp)
            else:
                aug_sp = sp
            selected_sps.append(aug_sp)
            selected_labs.append(lab)
        selected_sps = torch.stack(selected_sps, dim=0)
        selected_labs = torch.tensor(selected_labs, dtype=torch.long)
        return selected_sps, selected_labs

    def shuffle_data(self):
        random.shuffle(self.data)

    def is_empty(self):
        if len(self.data) == 0:
            return True
        else:
            return False

    def dump_data(self, task_id):
        dump_file = os.path.join(self.local_path, 'buffer_data' + str(task_id) + '.pkl')
        with open(dump_file, 'wb') as fw:
            pickle.dump(self.data, fw)
