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
from coreset_selection import q_vendi


def random_select(id2cnt, select_size):
    selected_id2prob = {}
    all_ids = list(id2cnt.keys())
    selected_ids = set(random.sample(all_ids, select_size))
    for d_id in selected_ids:
        selected_id2prob[d_id] = 1.0
    return selected_ids, selected_id2prob


def _select_by_qvendi(loss_diffs, id2features, id2pos, rand_data, incremental_size, 
                      class_sizes, id2logits, loss_params):
    """
    Select samples using Quality-Weighted Vendi Score.
    Quality = loss_diff (reducible loss)
    Diversity = computed from feature similarity
    """
    # prepare data structures
    all_ids = list(loss_diffs.keys())
    n = len(all_ids)
    
    # build feature matrix and quality scores
    features = np.array([id2features[did] for did in all_ids])
    quality_scores = np.array([loss_diffs[did] for did in all_ids])
    
    # normalize quality scores to be positive (add min if negative)
    min_quality = np.min(quality_scores)
    if min_quality < 0:
        quality_scores = quality_scores - min_quality + 1e-6
    else:
        quality_scores = quality_scores + 1e-6  # ensure positive
    
    # compute similarity matrix using cosine similarity
    # normalize features
    features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    K = np.dot(features_norm, features_norm.T)  # cosine similarity
    
    # greedy selection to maximize q_vendi score
    selected_indices = []
    selected_ids = []
    remaining_indices = list(range(n))
    
    # handle class balance constraints
    class_cnt = {}
    id2class = {}
    if class_sizes is not None:
        for ci in class_sizes.keys():
            class_cnt[ci] = 0
        for idx, did in enumerate(all_ids):
            pos = id2pos[did]
            di = rand_data[pos]
            lab = int(di[2])
            id2class[idx] = lab
    
    while len(selected_indices) < incremental_size and len(remaining_indices) > 0:
        best_idx = None
        best_qvs = -float('inf')
        
        for idx in remaining_indices:
            # check class balance constraint
            if class_sizes is not None:
                lab = id2class[idx]
                if class_cnt[lab] >= class_sizes[lab]:
                    continue
            
            # compute q_vendi with this sample added
            test_indices = selected_indices + [idx]
            K_subset = K[np.ix_(test_indices, test_indices)]
            quality_subset = quality_scores[test_indices]
            
            qvs = q_vendi.score_from_kernel_matrix(K_subset, quality_subset)
            
            if qvs > best_qvs:
                best_qvs = qvs
                best_idx = idx
        
        if best_idx is not None:
            selected_indices.append(best_idx)
            selected_ids.append(all_ids[best_idx])
            remaining_indices.remove(best_idx)
            
            if class_sizes is not None:
                lab = id2class[best_idx]
                class_cnt[lab] += 1
        else:
            break
    
    # build selected_data from selected_ids
    selected_data = []
    id2loss_dif = {}
    for did in selected_ids:
        pos = id2pos[did]
        di = rand_data[pos]
        new_di = copy.deepcopy(di)
        if loss_params['mse_factor'] > 0 and len(di) < 4:
            if did in id2logits:
                new_di.append(id2logits[did])
        selected_data.append(new_di)
        id2loss_dif[did] = loss_diffs[did]
    
    return selected_data, id2loss_dif


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
                        class_sizes=None, use_qvendi=False):
    status = model.training
    model.eval()
    if on_cuda:
        model.cuda()
    loss_fn = loss_functions.CompliedLoss(
        ce_factor=loss_params['ce_factor'], mse_factor=loss_params['mse_factor'], reduction='none')
    loss_diffs = {}
    id2pos = {}
    id2logits = {}
    id2features = {}  # store features for similarity computation
    batch_ids = []
    batch_sps = []
    batch_labs = []
    batch_logits = []
    
    # hook to extract features from the model (penultimate layer)
    features_dict = {}
    def hook_fn(module, input, output):
        features_dict['features'] = output.detach()
    
    # register hook on the layer before the final classifier
    hook_handle = None
    if use_qvendi:
        # strategy: hook the penultimate layer based on model architecture
        # try multiple strategies to find the right layer
        
        # strategy 1: models with embed() method (ConvNet, FNNet, ResNet)
        if hasattr(model, 'embed'):
            # for ConvNet: hook fc1 (the penultimate layer before fc2)
            if hasattr(model, 'fc1') and hasattr(model, 'fc2'):
                hook_handle = model.fc1.register_forward_hook(hook_fn)
            # for FNNet: hook fc2 (the penultimate layer before fc3)
            elif hasattr(model, 'fc2') and hasattr(model, 'fc3'):
                hook_handle = model.fc2.register_forward_hook(hook_fn)
            # for ResNet: hook layer4 (the last conv layer before pooling and linear)
            elif hasattr(model, 'layer4') and hasattr(model, 'linear'):
                hook_handle = model.layer4.register_forward_hook(hook_fn)
            # for other models with avgpool: look for avgpool or the last layer
            else:
                for name, module in model.named_modules():
                    if 'avgpool' in name or 'pool' in name:
                        hook_handle = module.register_forward_hook(hook_fn)
                        break
        
        # strategy 2: models with classifier attribute (ResNetDER, etc.)
        elif hasattr(model, 'classifier'):
            for name, module in model.named_modules():
                if 'avgpool' in name or 'pool' in name or 'flatten' in name:
                    hook_handle = module.register_forward_hook(hook_fn)
                    break
        
        # strategy 3: look for common penultimate layer names
        if hook_handle is None:
            for name, module in model.named_modules():
                if any(key in name.lower() for key in ['avgpool', 'pool', 'fc1', 'penultimate']):
                    hook_handle = module.register_forward_hook(hook_fn)
                    break
        
        # warn if hook registration failed
        if hook_handle is None:
            print("WARNING: Q-Vendi enabled but could not register feature extraction hook!")
            print(f"Model type: {type(model).__name__}")
            print("Available modules:", [name for name, _ in model.named_modules()])
    
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
                
                # forward pass
                output = model(sps)
                loss = loss_fn(x=output, y=labs, logits=lab_logits)
                loss = loss.clone().detach()
                if on_cuda:
                    loss = loss.cpu()
                loss = loss.numpy()
                
                # extract features if using q_vendi
                if use_qvendi and 'features' in features_dict:
                    batch_features = features_dict['features']
                    if on_cuda:
                        batch_features = batch_features.cpu()
                    batch_features = batch_features.numpy()
                    # flatten features if needed
                    if len(batch_features.shape) > 2:
                        batch_features = batch_features.reshape(batch_features.shape[0], -1)
                
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
                    if use_qvendi and 'features' in features_dict:
                        id2features[did] = batch_features[j]
                
                batch_ids.clear()
                batch_sps.clear()
                batch_labs.clear()
                batch_logits.clear()
                del lab_logits
                features_dict.clear()
    
    if hook_handle is not None:
        hook_handle.remove()
    
    # selection based on q_vendi or original loss_diff
    if use_qvendi and len(id2features) > 0:
        print(f"Using Q-Vendi selection with {len(id2features)} samples")
        selected_data, id2loss_dif = _select_by_qvendi(
            loss_diffs=loss_diffs,
            id2features=id2features,
            id2pos=id2pos,
            rand_data=rand_data,
            incremental_size=incremental_size,
            class_sizes=class_sizes,
            id2logits=id2logits,
            loss_params=loss_params
        )
    else:
        if use_qvendi:
            print(f"WARNING: Q-Vendi requested but falling back to loss_diff selection (features extracted: {len(id2features)})")
        # original selection by loss_diff
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
