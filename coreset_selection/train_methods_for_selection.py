# -*-coding:utf8-*-

import torch
import numpy as np
import os
import copy
import pickle

from functions import loss_functions


def train_model(local_path, model, train_loader, train_params, eval_loader, eval_mode, verbose=True,
                load_best=False, eval_steps=10):
    if 'log_steps' in train_params:
        log_steps = train_params['log_steps']
    else:
        log_steps = 100
    if 'loss_params' in train_params:
        loss_fn = loss_functions.CompliedLoss(
            ce_factor=train_params['loss_params']['ce_factor'],
            mse_factor=train_params['loss_params']['mse_factor']
        )
        loss_params = train_params['loss_params']
    else:
        loss_fn = torch.nn.CrossEntropyLoss()
        loss_params = None
    if train_params['use_cuda']:
        model = model.cuda()
    if 'opt_type' in train_params:
        if train_params['opt_type'] == 'adam':
            opt = torch.optim.Adam(lr=train_params['lr'], params=model.parameters())
        else:
            opt = torch.optim.SGD(lr=train_params['lr'], params=model.parameters())
    else:
        opt = torch.optim.SGD(lr=train_params['lr'], params=model.parameters())
    best_score = None
    bad_cnt = 0
    # train model
    step = 0
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
        opt.step()
        if verbose and step % log_steps == 0:
            print('loss at step:', step, 'is', loss.item())
        if step % eval_steps == 0 or step == train_params['steps'] - 1:
            # evaluation
            score = eval_model(
                model=model,
                eval_loader=eval_loader,
                eval_mode=eval_mode,
                on_cuda=train_params['use_cuda'],
                loss_params=loss_params
            )
            if verbose:
                print('score in step', step, 'is:', score)
            if best_score is None or score > best_score:
                best_score = score
                bad_cnt = 0
                save_model(local_path=local_path, model=model, on_cuda=train_params['use_cuda'])
            else:
                bad_cnt += 1
            if 'early_stop' in train_params and bad_cnt == train_params['early_stop']:
                print('\tearly stop at step:', step)
                break
        step += 1
        if step == train_params['steps']:
            break
    if load_best and train_params['early_stop'] > 0:
        model = load_model(local_path=local_path)
    if verbose:
        print('\tbest score is:', best_score)
    if train_params['use_cuda']:
        model = model.cpu()
    clear_temp_model(local_path=local_path)
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


def eval_model(model, eval_loader, eval_mode, loss_params, on_cuda=True):
    if eval_mode == 'acc':
        score = compute_accuracy(
            model=model, eval_loader=eval_loader, on_cuda=on_cuda
        )
    elif eval_mode == 'avg_loss':
        score = compute_avg_loss(
            eval_loader=eval_loader, model=model, loss_params=loss_params, on_cuda=on_cuda
        )
    elif eval_mode == 'loss_var':
        score = compute_loss_var(
            eval_loader=eval_loader, model=model, loss_params=loss_params, on_cuda=on_cuda
        )
    else:
        score = None
    return score


def compute_accuracy(model, eval_loader, on_cuda=False):
    with torch.no_grad():
        total_num = 0
        right_num = 0
        for e_data in eval_loader:
            if len(e_data) == 2:
                sp, lab = e_data
            elif len(e_data) == 4:
                d_id, sp, lab, logit = e_data
            else:
                d_id, sp, lab = e_data
            if on_cuda:
                sp = sp.cuda()
            out = model(sp).clone().detach()
            if on_cuda:
                out = out.cpu()
            out = out.numpy()
            lab = lab.numpy()
            for j in range(out.shape[0]):
                pred = np.argmax(out[j, :])
                if int(lab[j]) == int(pred):
                    right_num += 1
                total_num += 1
        acc = right_num / total_num
    return acc


def compute_avg_loss(eval_loader, model, loss_params=None, on_cuda=True):
    status = model.training
    model.eval()
    if loss_params is None:
        loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    else:
        loss_fn = loss_functions.CompliedLoss(
            ce_factor=loss_params['ce_factor'],
            mse_factor=loss_params['mse_factor'],
            reduction='sum'
        )
    total_loss = 0
    cnt = 0
    with torch.no_grad():
        for data in eval_loader:
            if len(data) == 3:
                d_id, sp, lab = data
                logit = None
            elif len(data) == 4:
                d_id, sp, lab, logit = data
            else:
                sp, lab = data
                logit = None
            if on_cuda:
                sp = sp.cuda()
                lab = lab.cuda()
                if logit is not None:
                    logit = logit.cuda()
            if loss_params is None:
                loss = loss_fn(model(sp), lab).clone().detach()
            else:
                loss = loss_fn(model(sp), lab, logit).clone().detach()
            if on_cuda:
                loss = loss.cpu()
            loss = loss.numpy()
            total_loss += loss
            cnt += sp.shape[0]
    avg_loss = -1 * (total_loss / cnt)
    model.train(status)
    return avg_loss


def compute_loss_var(eval_loader, model, loss_params=None, on_cuda=True):
    status = model.training
    model.eval()
    if loss_params is None:
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    else:
        loss_fn = loss_functions.CompliedLoss(
            ce_factor=loss_params['ce_factor'],
            mse_factor=loss_params['mse_factor'],
            reduction='none'
        )
    losses = []
    with torch.no_grad():
        for data in eval_loader:
            if len(data) == 3:
                d_id, sp, lab = data
                logit = None
            elif len(data) == 4:
                d_id, sp, lab, logit = data
            else:
                sp, lab = data
                logit = None
            if on_cuda:
                sp = sp.cuda()
                lab = lab.cuda()
                if logit is not None:
                    logit = logit.cuda()
            if loss_params is None:
                loss = loss_fn(model(sp), lab).clone().detach()
            else:
                loss = loss_fn(model(sp), lab, logit).clone().detach()
            if on_cuda:
                loss = loss.cpu()
            loss = loss.numpy()
            for i in range(sp.shape[0]):
                losses.append(loss[i])
    model.train(status)
    loss_var = np.var(losses)
    return loss_var
