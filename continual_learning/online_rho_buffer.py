# -*-coding:utf8-*-

import copy
import os
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import single_task_dataset
import utils
from coreset_selection import train_methods_for_selection


class OnlineRhoBuffer(object):
    def __init__(self, local_path, buffer_size, selection_params, model_params, use_cuda, transforms, seed,
                 remove_mode='random', add_mode='rho_loss'):
        """
        this buffer requires providing data id for selection.
        """
        self.local_path = local_path
        if not os.path.exists(self.local_path):
            os.makedirs(self.local_path)
        self.buffer_size = buffer_size
        self.selection_params = selection_params
        self.model_params = model_params
        self.use_cuda = use_cuda
        self.transforms = transforms
        self.seed = seed
        self.remove_mode = remove_mode
        self.add_mode = add_mode
        # parameters for selection
        self.id2loss = {}
        self.id2logit = {}  # to keep logit all current data, and we use this logit to train buffer model
        self.id2pos = {}
        self.data = []
        self.seen_samples = 0
        self.cur_model = utils.build_model(model_params=model_params)
        self.cur_model.eval()
        if self.use_cuda:
            self.cur_model.cuda()
        self.seen_tasks = 0
        self.id2task = {}
        self.upd_cnt = 0

    def update_buffer(self, d_ids, sps, labs, logits, ce_loss):
        if logits is not None:
            # update id2logit
            logits_numpy = logits.numpy()
            for i in range(len(sps)):
                d_id = int(d_ids[i])
                self.id2logit[d_id] = logits_numpy[i, :]
        if self.add_mode == 'rho_loss':
            self.update_buffer_rho(
                d_ids=d_ids, sps=sps, labs=labs, logits=logits, ce_loss=ce_loss)
        elif self.add_mode == 'random':
            for i, sp in enumerate(sps):
                if logits is None:
                    self.update_buffer_random(sp=sp, lab=labs[i], logit=None)
                else:
                    self.update_buffer_random(sp=sp, lab=labs[i], logit=logits.numpy()[i, :])
        else:
            raise ValueError('Invalid update mode')

    def update_buffer_rho(self, d_ids, sps, labs, logits, ce_loss):
        """
        This function computes loss difference to select which sample is most worthy to store.
        :param d_ids: batch data ids
        :param sps: not augmented samples in batch
        :param labs: labels in batch
        :param logits: logits tensor of current model in batch
        :param ce_loss: vector of cross-entropy losses in batch
        :return:
        """
        loss_diffs = []
        loss_params = self.selection_params['slt_params']
        cur_losses = self.compute_batch_state(sps=sps, labs=labs, logits=logits)
        # compute loss difference
        for i in range(len(sps)):
            loss_diff = cur_losses[i] - loss_params['ce_factor'] * ce_loss[i]
            loss_diffs.append(loss_diff)
        # normalize scores to 1.0
        softmax_fn = torch.nn.Softmax(dim=-1)
        normed_scales = softmax_fn(torch.tensor(loss_diffs, dtype=torch.float32)).numpy() * len(sps)
        # update sample
        if logits is not None:
            logits_numpy = logits.numpy()
        else:
            logits_numpy = None
        num_upd = 0
        for i in range(len(sps)):
            if self.seen_samples < self.buffer_size:
                if int(d_ids[i]) in self.id2pos:  # prevent duplicated data
                    if logits is not None:  # only update logit
                        pos = self.id2pos[int(d_ids[i])]
                        self.data[pos][3] = logits_numpy[i, :]
                else:
                    self.id2pos[int(d_ids[i])] = len(self.data)
                    if logits is None:
                        self.data.append([int(d_ids[i]), sps[i], labs[i]])
                    else:
                        self.data.append([int(d_ids[i]), sps[i], labs[i], logits_numpy[i, :]])
                num_upd += 1
            else:
                # compute different possibility
                rand = np.random.randint(0, self.seen_samples + 1)
                if rand < self.buffer_size * normed_scales[i]:
                    if int(d_ids[i]) in self.id2pos:  # prevent duplicated data
                        if logits is not None:  # only update logit
                            pos = self.id2pos[int(d_ids[i])]
                            self.data[pos][3] = logits_numpy[i, :]
                    else:
                        # randomly remove existing samples
                        r_pos = np.random.randint(0, self.buffer_size)
                        ori_id = self.data[r_pos][0]
                        del self.id2pos[ori_id]
                        self.id2pos[int(d_ids[i])] = r_pos
                        if logits_numpy is None:
                            self.data[r_pos] = [int(d_ids[i]), sps[i], labs[i]]
                        else:
                            self.data[r_pos] = [int(d_ids[i]), sps[i], labs[i], logits_numpy[i, :]]
                    num_upd += 1
            self.seen_samples += 1
        self.upd_cnt += num_upd
        if self.upd_cnt >= max(
                (self.buffer_size / (self.seen_tasks + 1)) / self.selection_params['selection_steps'], 1):
            self.train_buffer_model()
            self.upd_cnt = 0
        return num_upd

    def compute_batch_state(self, sps, labs, logits=None):
        aug_sps = []
        for sp in sps:
            aug_sp = self.transforms(sp)
            aug_sps.append(aug_sp)
        aug_sps = torch.stack(aug_sps, dim=0)
        labs = torch.tensor(labs, dtype=torch.long)
        lab_logits = logits
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        kd_loss_fn = torch.nn.MSELoss(reduction='none')
        loss_params = self.selection_params['slt_params']
        if self.use_cuda:
            aug_sps = aug_sps.cuda()
            labs = labs.cuda()
            if lab_logits is not None:
                lab_logits = lab_logits.cuda()
        with torch.no_grad():
            out_logits = self.cur_model(aug_sps)
            ce_losses = loss_fn(out_logits, labs)
            if lab_logits is not None:
                mse_losses = torch.mean(kd_loss_fn(out_logits, lab_logits), dim=-1)
            else:
                mse_losses = 0
            loss = loss_params['ce_factor'] * ce_losses + loss_params['mse_factor'] * mse_losses
        if self.use_cuda:
            loss = loss.cpu()
        loss = loss.clone().detach().numpy()
        return loss

    def update_buffer_random(self, sp, lab, logit=None):
        upd = False
        if self.seen_samples < self.buffer_size:
            if logit is None:
                self.data.append([-1, sp, lab])
            else:
                self.data.append([-1, sp, lab, logit])
        else:
            rand = np.random.randint(0, self.seen_samples + 1)
            if rand < self.buffer_size:
                if logit is None:
                    self.data[rand] = [-1, sp, lab]
                else:
                    self.data[rand] = [-1, sp, lab, logit]
                upd = True
        self.seen_samples += 1
        return upd

    def train_buffer_model(self):
        # extract current data
        cur_data = []
        for di in self.data:
            d_id = int(di[0])
            if d_id not in self.id2task:
                new_di = copy.deepcopy(di)
                if len(new_di) > 3:  # use new logit to train buffer model
                    new_di[3] = self.id2logit[d_id]
                cur_data.append(new_di)
        # make train loader
        temp_train_file = os.path.join(self.local_path, 'temp_train.pkl')
        with open(temp_train_file, 'wb') as fw:
            for di in cur_data:
                pickle.dump(di, fw)
        temp_train_dataset = single_task_dataset.RandomDataset(
            seed=self.seed,
            data_path=temp_train_file,
            transforms=self.transforms,  # the transforms can be altered here
            extra_data=None
        )
        temp_train_loader = DataLoader(temp_train_dataset, batch_size=32, drop_last=False)
        # init model and train model
        init_model = utils.build_model(model_params=self.model_params)
        trained_model = train_methods_for_selection.train_model(
            local_path=self.local_path,
            model=init_model,
            train_loader=temp_train_loader,
            train_params=self.selection_params['cur_train_params'],
            eval_loader=None,
            eval_mode='none',
            verbose=False,
            load_best=False,
            eval_steps=10
        )
        os.remove(temp_train_file)
        if self.use_cuda:
            trained_model.cuda()
        self.cur_model = trained_model
        self.cur_model.eval()

    def get_sub_data(self, data_size):
        inds = np.random.choice(
            min(self.seen_samples, len(self.data)),
            size=min(data_size, len(self.data)),
            replace=False
        )
        selected_sps = []
        selected_labs = []
        selected_logits = []
        for idx in inds:
            di = self.data[idx]
            if len(di) == 3:
                d_id, sp, lab = di
            elif len(di) == 4:
                d_id, sp, lab, logit = di
                selected_logits.append(torch.tensor(logit, dtype=torch.float32))
            else:
                raise ValueError('Invalid data length')
            aug_sp = self.transforms(sp)
            selected_sps.append(aug_sp)
            selected_labs.append(lab)
        selected_sps = torch.stack(selected_sps, dim=0)
        selected_labs = torch.tensor(selected_labs, dtype=torch.long)
        if len(selected_logits) > 0:
            selected_logits = torch.stack(selected_logits, dim=0)
            return selected_sps, selected_labs, selected_logits
        else:
            return selected_sps, selected_labs

    def is_empty(self):
        if len(self.data) == 0:
            return True
        else:
            return False

    def next_task(self):
        self.id2loss.clear()
        self.id2logit.clear()
        for di in self.data:
            d_id = int(di[0])
            if d_id not in self.id2task:
                self.id2task[d_id] = self.seen_tasks
        self.seen_tasks += 1
        self.cur_model = utils.build_model(model_params=self.model_params)
        self.cur_model.eval()
        if self.use_cuda:
            self.cur_model.cuda()
        self.upd_cnt = 0

    def dump_buffer(self, task_id):
        dump_file = os.path.join(self.local_path, 'buffer_data' + str(task_id) + '.pkl')
        with open(dump_file, 'wb') as fw:
            pickle.dump(self.data, fw)
