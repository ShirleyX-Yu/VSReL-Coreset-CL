# -*-coding:utf8-*-

import copy
import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import torchvision

from continual_learning import coreset_buffer
import utils
from dataset import single_task_dataset


class ContinualRunner(object):
    def __init__(self, local_path, model_params, transforms, train_params, selection_params, use_cuda,
                 task_dic, buffer_size, seed, replay_mode='full', selection_transforms=None,
                 extra_data_mode=None, buffer_type='coreset'):
        # make parameters
        self.local_path = local_path
        if not os.path.exists(self.local_path):
            os.makedirs(self.local_path)
        self.model_params = model_params
        self.transforms = transforms
        self.selection_params = selection_params
        self.seed = seed
        self.use_cuda = use_cuda
        self.train_params = train_params
        self.task_dic = task_dic
        self.buffer_size = buffer_size
        self.replay_mode = replay_mode
        self.extra_data_mode = extra_data_mode
        self.buffer_type = buffer_type
        # build model
        self.model = utils.build_model(model_params=model_params)
        # build buffer:
        if self.buffer_type == 'coreset':
            self.buffer = coreset_buffer.CoresetBuffer(
                local_path=os.path.join(self.local_path, 'buffer'),
                model_params=self.model_params,
                transforms=self.transforms,
                selection_params=self.selection_params,
                use_cuda=self.use_cuda,
                buffer_size=buffer_size,
                task_dic=self.task_dic,
                seed=self.seed,
                selection_transforms=selection_transforms,
                extra_data_mode=self.extra_data_mode
            )
        elif self.buffer_type == 'uniform':
            self.buffer = coreset_buffer.UniformBuffer(
                local_path=os.path.join(self.local_path, 'buffer'),
                transforms=self.transforms,
                buffer_size=self.buffer_size,
                use_cuda=self.use_cuda,
                seed=self.seed
            )
        else:
            raise ValueError('Invalid buffer type')
        self.to_pil = torchvision.transforms.ToPILImage()
        self.seen_tasks = 0
        self.results = []
        self.mask_results = []
        self.task_cnts = []

    def train_single_task(self, train_loader, eval_loaders, verbose=True, do_evaluation=True):
        self.model.train()
        if self.train_params['use_cuda']:
            self.model.cuda()
        # make loss function
        loss_fn = torch.nn.CrossEntropyLoss()
        # make optimizer
        if self.train_params['opt_type'] == 'adam':
            opt = torch.optim.Adam(lr=self.train_params['lr'], params=self.model.parameters())
        else:
            opt = torch.optim.SGD(lr=self.train_params['lr'], params=self.model.parameters())
        # make hyper parameters
        alpha = torch.tensor(self.train_params['alpha'], dtype=torch.float32, requires_grad=False)
        if self.train_params['use_cuda']:
            alpha = alpha.cuda()
        if 'beta' in self.train_params:
            beta = torch.tensor(self.train_params['beta'], dtype=torch.float32, requires_grad=False)
            if self.train_params['use_cuda']:
                beta = beta.cuda()
            kd_loss_fn = torch.nn.MSELoss()
        else:
            beta = 0
            kd_loss_fn = None
        for i in range(self.train_params['epochs']):
            step = 0
            for data in train_loader:
                aug_sp, lab = data
                if self.use_cuda:
                    aug_sp = aug_sp.cuda()
                    lab = lab.cuda()
                if not self.buffer.is_empty():
                    loss = loss_fn(self.model(aug_sp), lab)
                    total_pre_loss = []
                    total_kd_loss = []
                    if self.replay_mode == 'full':
                        for buffer_data in self.buffer.get_data():
                            if len(buffer_data) == 3:
                                b_sp, b_lab, b_logit = buffer_data
                            else:
                                b_sp, b_lab = buffer_data
                                b_logit = None
                            if self.train_params['use_cuda']:
                                b_sp = b_sp.cuda()
                                b_lab = b_lab.cuda()
                                if b_logit is not None:
                                    b_logit = b_logit.cuda()
                            b_out = self.model(b_sp)
                            pre_loss = loss_fn(b_out, b_lab)
                            total_pre_loss.append(pre_loss.item())
                            if kd_loss_fn is not None:
                                kd_loss = kd_loss_fn(b_out, b_logit)
                                total_kd_loss.append(kd_loss.item())
                            else:
                                kd_loss = 0
                                total_kd_loss.append(0)
                            loss = loss + alpha * pre_loss + beta * kd_loss
                        if verbose and step % 100 == 0:
                            print('loss at step ', step, 'is:', loss.item())
                            print('previous loss at step ', step, 'is:', np.mean(total_pre_loss), len(total_pre_loss))
                            if kd_loss_fn is not None:
                                print('kd loss is:', np.mean(total_kd_loss), len(total_kd_loss))
                    else:
                        buffer_data = self.buffer.get_sub_data(size=self.train_params['mem_batch_size'])
                        if len(buffer_data) == 3:
                            b_sp, b_lab, b_logit = buffer_data
                        else:
                            b_sp, b_lab = buffer_data
                            b_logit = None
                        if self.train_params['use_cuda']:
                            b_sp = b_sp.cuda()
                            b_lab = b_lab.cuda()
                            if b_logit is not None:
                                b_logit = b_logit.cuda()
                        b_out = self.model(b_sp)
                        pre_loss = loss_fn(b_out, b_lab)
                        if kd_loss_fn is not None:
                            kd_loss = kd_loss_fn(b_out, b_logit)
                        else:
                            kd_loss = 0
                        loss = loss + alpha * pre_loss + beta * kd_loss
                        if verbose and step % 100 == 0:
                            print('loss at step ', step, 'is:', loss.item())
                            print('previous loss at step ', step, 'is:', pre_loss.item())
                            if kd_loss_fn is not None:
                                print('kd loss is:', kd_loss.item())
                else:
                    out_logits = self.model(aug_sp)
                    loss = loss_fn(out_logits, lab)
                    if verbose and step % 100 == 0:
                        print('loss at step ', step, 'is:', loss.item())
                opt.zero_grad()
                loss.backward()
                opt.step()
                step += 1
            print('finish training epoch:', i)
            if do_evaluation and i % 10 == 0:
                accs, losses = self.evaluate_model(eval_loaders=eval_loaders, on_cuda=self.use_cuda)
                print('\taccuracy on test is:', np.mean(accs), accs, losses)
                if len(accs) > 1:
                    print('\tprevious tasks accuracy is:', np.mean(accs[:-1]))
        if self.train_params['use_cuda']:
            self.model.cpu()
        del alpha
        accs, losses = self.evaluate_model(eval_loaders=eval_loaders, on_cuda=False)
        print('\tlosses on testset is:', losses)
        self.results.append(accs)
        return accs

    def evaluate_model(self, eval_loaders, on_cuda=False):
        status = self.model.training
        self.model.eval()
        accs = []
        eval_loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
        losses = []
        with torch.no_grad():
            for i in range(self.seen_tasks + 1):
                loss = 0
                correct = 0
                for data, target in eval_loaders[i]:
                    if on_cuda:
                        data, target = data.cuda(), target.cuda()
                    output = self.model(data)
                    loss += eval_loss_fn(output, target).cpu().item()
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                avg_acc = 100. * correct / len(eval_loaders[i].dataset)
                avg_loss = loss / len(eval_loaders[i].dataset)
                accs.append(avg_acc)
                losses.append(avg_loss)
        self.model.train(status)
        return accs, losses

    def update_buffer(self, full_train_loader, sub_loader, next_loader=None):
        # make current x and y
        x, y = next(iter(full_train_loader))
        full_cur_x = x.numpy()
        full_cur_y = y.numpy()
        cur_x, cur_y = next(iter(sub_loader))
        self.task_cnts.append(cur_x.shape[0])
        if 'beta' in self.train_params:
            temp_data = []
            for i in range(cur_x.shape[0]):
                sp = self.to_pil(cur_x[i, :])
                lab = int(cur_y[i])
                temp_data.append([sp, lab])
            temp_dataset = single_task_dataset.SimpleDataset(
                data=temp_data,
                transforms=self.transforms
            )
            temp_loader = DataLoader(
                temp_dataset, batch_size=self.train_params['batch_size'], drop_last=False, shuffle=False)
            cur_id2logit = utils.compute_id2logit(
                data_loader=temp_loader,
                ref_model=copy.deepcopy(self.model),
                aug_iters=1,
                use_cuda=True
            )
        else:
            cur_id2logit = None
        if self.extra_data_mode is not None and 'next_task' in self.extra_data_mode and next_loader is not None:
            next_x, next_y = next(iter(next_loader))
            next_x = next_x.numpy()
            next_y = next_y.numpy()
        else:
            next_x = None
            next_y = None
        cur_x = cur_x.numpy()
        cur_y = cur_y.numpy()
        if isinstance(self.buffer, coreset_buffer.CoresetBuffer):
            self.buffer.update_buffer(
                task_cnts=self.task_cnts,
                task_id=self.seen_tasks,
                cur_x=cur_x,
                cur_y=cur_y,
                full_cur_x=full_cur_x,
                full_cur_y=full_cur_y,
                cur_id2logit=cur_id2logit,
                next_x=next_x,
                next_y=next_y
            )
        elif isinstance(self.buffer, coreset_buffer.UniformBuffer):
            self.buffer.update_buffer(
                task_cnts=self.task_cnts,
                task_id=self.seen_tasks,
                cur_x=cur_x,
                cur_y=cur_y
            )
        else:
            raise ValueError('Invalid buffer type')

    def next_task(self, dump_buffer=False):
        if dump_buffer:
            self.buffer.dump_data(task_id=self.seen_tasks)
        self.seen_tasks += 1
