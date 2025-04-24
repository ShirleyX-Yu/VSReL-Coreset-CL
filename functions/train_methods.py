# -*-coding:utf8-*-
import pickle

import torch
import numpy as np
import os
import copy

import utils
from functions import loss_functions


def train_model(local_path, model, train_loader, eval_loader, epochs, train_params, verbose=True, save_ckpt=False,
                load_best=False, weight_decay=0, log_file=None):
    if 'log_steps' in train_params:
        log_steps = train_params['log_steps']
    else:
        log_steps = 100
    if 'loss_params' in train_params:
        loss_fn = loss_functions.CompliedLoss(
            ce_factor=train_params['loss_params']['ce_factor'],
            mse_factor=train_params['loss_params']['mse_factor']
        )
    else:
        loss_fn = torch.nn.CrossEntropyLoss()
    if train_params['use_cuda']:
        model = model.cuda()
    if 'opt_type' in train_params:
        if train_params['opt_type'] == 'adam':
            opt = torch.optim.Adam(lr=train_params['lr'], weight_decay=weight_decay, params=model.parameters())
        else:
            opt = torch.optim.SGD(lr=train_params['lr'], weight_decay=weight_decay, params=model.parameters())
    else:
        opt = torch.optim.SGD(lr=train_params['lr'], weight_decay=weight_decay, params=model.parameters())
    if 'scheduler_type' in train_params:
        if train_params['scheduler_type'] == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=train_params['epochs'])
        elif train_params['scheduler_type'] == 'ReduceLROnPlateau':
            # if we need to re-init model, we reduce LR to ensure convergence
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt,
                factor=train_params['scheduler_param']['factor'],  # new-lr = factor * lr
                patience=train_params['scheduler_param']['patience'],  # how many time if there is no improvement
                min_lr=train_params['scheduler_param']['min_lr'],  # minimal lr
                verbose=verbose  # print lr change
            )
        else:
            scheduler = None
    else:
        scheduler = None
    best_acc = None
    bad_cnt = 0
    losses = []
    for i in range(epochs):
        # train model
        step = 0
        total_loss = 0
        for data in train_loader:
            if len(data) == 2:
                sp, lab = data
                ref_logits = None
            elif len(data) == 4:
                d_id, sp, lab, ref_logits = data
            else:
                d_id, sp, lab = data
                ref_logits = None
            if train_params['use_cuda']:
                sp = sp.cuda()
                lab = lab.cuda()
                if ref_logits is not None:
                    ref_logits = ref_logits.cuda()
            out = model(sp)
            if 'loss_params' in train_params:
                loss = loss_fn(x=out, y=lab, logits=ref_logits)
            else:
                loss = loss_fn(out, lab)
            opt.zero_grad()
            loss.backward()
            if 'grad_max_norm' in train_params and train_params['grad_max_norm'] is not None:
                # add gradient clipping for optimization
                torch.nn.utils.clip_grad_norm_(
                    parameters=model.parameters(),
                    max_norm=train_params['grad_max_norm'],
                    norm_type=2
                )
            opt.step()
            total_loss += loss.item()
            if step % log_steps == 0:
                if verbose:
                    print('loss at step:', step, 'is', loss.item())
                if log_file is not None:
                    losses.append(float(loss.item()))
            step += 1
        avg_loss = total_loss / step
        if i % 10 == 0 or i == epochs - 1:
            if save_ckpt:
                save_name = 'model_' + str(i) + '.pkl'
                save_model(
                    local_path=local_path,
                    model=model,
                    on_cuda=train_params['use_cuda'],
                    save_name=save_name
                )
            # evaluation
            if eval_loader is not None:
                acc = eval_model(
                    model=model,
                    on_cuda=train_params['use_cuda'],
                    eval_loader=eval_loader
                )
            else:
                acc = None
            if verbose:
                print('accuracy in epoch', i, 'is:', acc)
            if best_acc is None or acc > best_acc:
                best_acc = acc
                bad_cnt = 0
                save_model(local_path=local_path, model=model, on_cuda=train_params['use_cuda'])
            else:
                bad_cnt += 1
            if 'early_stop' in train_params and bad_cnt == train_params['early_stop']:
                print('\tearly stop at epoch:', i)
                break
        if scheduler is not None:
            if train_params['scheduler_type'] == 'CosineAnnealingLR':
                scheduler.step()
            elif train_params['scheduler_type'] == 'ReduceLROnPlateau':
                scheduler.step(avg_loss)
        else:
            pass
        if i < epochs - 1 and hasattr(train_loader.dataset, 'shuffle_dataset'):
            train_loader.dataset.shuffle_dataset()
    if save_ckpt:
        save_name = 'model_' + str(epochs) + '.pkl'
        save_model(
            local_path=local_path,
            model=model,
            on_cuda=train_params['use_cuda'],
            save_name=save_name
        )
    if eval_loader is not None and load_best and train_params['early_stop'] > 0:
        model = load_model(local_path=local_path)
    if verbose:
        print('\tbest accuracy is:', best_acc)
    if train_params['use_cuda']:
        model = model.cpu()
    clear_temp_model(local_path=local_path)
    if hasattr(train_loader.dataset, 'remove_shuffle_file'):
        train_loader.dataset.remove_shuffle_file()
    # save loss
    if log_file is not None:
        with open(log_file, 'wb') as fw:
            pickle.dump(losses, fw)
    return model


def save_model(local_path, model, on_cuda, save_name=None):
    if save_name is None:
        out_model_file = os.path.join(local_path, 'best_model.pkl')
    else:
        out_model_file = os.path.join(local_path, save_name)
    saved_model = copy.deepcopy(model)
    if on_cuda:
        saved_model.cpu()
    torch.save(saved_model, out_model_file)


def load_model(local_path, save_name=None):
    if save_name is None:
        out_model_file = os.path.join(local_path, 'best_model.pkl')
    else:
        out_model_file = os.path.join(local_path, save_name)
    model = torch.load(out_model_file)
    return model


def clear_temp_model(local_path):
    temp_model_file = os.path.join(local_path, 'best_model.pkl')
    if os.path.exists(temp_model_file):
        os.remove(temp_model_file)


def eval_model(model, eval_loader, on_cuda=False, return_loss=False):
    status = model.training
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    with torch.no_grad():
        total_num = 0
        total_loss = 0
        right_num = 0
        for e_data in eval_loader:
            if len(e_data) == 2:
                sp, lab = e_data
            else:
                d_id, sp, lab = e_data
            if on_cuda:
                sp = sp.cuda()
                lab = lab.cuda()
            out = model(sp).clone().detach()
            loss = loss_fn(out, lab)
            if on_cuda:
                out = out.cpu()
                lab = lab.cpu()
                loss = loss.cpu()
            out = out.numpy()
            lab = lab.numpy()
            loss = loss.clone().detach().numpy()
            total_loss += loss
            for j in range(out.shape[0]):
                pred = np.argmax(out[j, :])
                if int(lab[j]) == int(pred):
                    right_num += 1
                total_num += 1
        acc = right_num / total_num
        avg_loss = total_loss / total_num
    model.train(status)
    if return_loss:
        return acc, avg_loss
    else:
        return acc


def eval_continual_model(model, eval_loader, task_dic, seen_tasks, on_cuda=False,
                         continual_setting='class-il', return_loss=False):
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    status = model.training
    model.eval()
    accs = []
    accs_mask_classes = []
    losses = []
    with torch.no_grad():
        for i in range(seen_tasks + 1):
            total_count = 0
            right_count = 0
            mask_right_count = 0
            other_classes = []
            total_loss = 0
            if continual_setting == 'class-il':
                for k in range(len(task_dic)):
                    if k != i:
                        other_classes += task_dic[k]
            eval_loader.dataset.set_cur_task(task_id=i)
            for data in eval_loader:
                sp, lab = data
                if on_cuda:
                    sp = sp.cuda()
                    lab = lab.cuda()
                out = model(sp)
                loss = loss_fn(out, lab)
                if on_cuda:
                    out = out.cpu()
                    lab = lab.cpu()
                    loss = loss.cpu()
                loss = loss.numpy()
                total_loss += np.sum(loss)
                out = out.clone().detach().numpy()
                lab = lab.numpy()
                for j in range(sp.shape[0]):
                    pred = np.argmax(out[j, :])
                    if continual_setting == 'class-il':
                        mask_pred = np.argmax(
                            utils.mask_classes(
                                out[j, :],
                                mask_ids=other_classes
                            )
                        )
                    else:
                        mask_pred = -1
                    if int(lab[j]) == int(pred):
                        right_count += 1
                    if int(lab[j]) == int(mask_pred):
                        mask_right_count += 1
                    total_count += 1
            acc = right_count / total_count
            accs.append(acc)
            mask_acc = mask_right_count / total_count
            accs_mask_classes.append(mask_acc)
            avg_loss = total_loss / total_count
            losses.append(avg_loss)
    model.train(status)
    if return_loss:
        return accs, accs_mask_classes, losses
    else:
        return accs, accs_mask_classes
