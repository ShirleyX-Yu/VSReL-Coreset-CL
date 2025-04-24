# -*-coding:utf8-*-

import os
import pickle
import numpy as np
import gc
import random
import torch
from torch.utils.data import IterableDataset
from torch.utils.data import Dataset
import torchvision


class PILDataset(IterableDataset):
    def __init__(self, data_path, local_path=None, transforms=None):
        self.local_path = local_path
        self.data_path = data_path
        if data_path.endswith('.pkl'):
            self.f_list = [data_path]
        else:
            self.f_list = [os.path.join(self.data_path, fi) for fi in os.listdir(self.data_path)]
        self.transforms = transforms
        # local parameters
        self.produce_id = False
        self.total_samples = 0
        self.lab2idx = {}
        self.count_data_size()

    def count_data_size(self):
        self.lab2idx.clear()
        sp_cnt = 0
        for fi in self.f_list:
            if not os.path.exists(fi):
                continue
            with open(fi, 'rb') as fr:
                while True:
                    try:
                        di = pickle.load(fr)
                        if len(di) == 3:
                            d_id, sp, lab = di
                        elif len(di) == 4:
                            d_id, sp, lab, ref_logit = di
                        else:
                            raise ValueError('Not enough values')
                        if int(lab) not in self.lab2idx:
                            self.lab2idx[int(lab)] = [int(d_id)]
                        else:
                            self.lab2idx[int(lab)].append(int(d_id))
                        sp_cnt += 1
                    except EOFError:
                        break
        self.total_samples = sp_cnt

    def get_class_ids(self, c):
        return self.lab2idx[c]

    def get_class_id_dic(self):
        return self.lab2idx

    def get_data_size(self):
        return self.total_samples

    def set_produce_id(self, produce_id):
        self.produce_id = produce_id

    def __iter__(self):
        for fi in self.f_list:
            with open(fi, 'rb') as fr:
                while True:
                    try:
                        data = pickle.load(fr)
                        if len(data) == 3:
                            d_id, sp, lab = data
                            ref_logit = None
                        elif len(data) == 4:
                            d_id, sp, lab, ref_logit = data
                        else:
                            raise ValueError('Not enough values')
                        if self.transforms is not None:
                            sp = self.transforms(sp)
                        if self.produce_id:
                            if ref_logit is not None:
                                yield d_id, sp, lab, ref_logit
                            else:
                                yield d_id, sp, lab
                        else:
                            if ref_logit is not None:
                                yield sp, lab, ref_logit
                            else:
                                yield sp, lab
                    except EOFError:
                        break

    def get_ori_samples_by_id(self, ids):
        selected_data = []
        for fi in self.f_list:
            with open(fi, 'rb') as fr:
                while True:
                    try:
                        data = pickle.load(fr)
                        d_id, sp, lab = data
                        if d_id in ids:
                            selected_data.append(data)
                    except EOFError:
                        break
        return selected_data

    def get_data_by_id(self, ids):
        id_set = set(ids)
        selected_data = []
        for fi in self.f_list:
            with open(fi, 'rb') as fr:
                while True:
                    try:
                        data = pickle.load(fr)
                        if len(data) > 2:
                            d_id = data[0]
                            if d_id in id_set:
                                selected_data.append(data)
                        else:
                            raise ValueError('No data-id in the dataset')
                    except EOFError:
                        break
        return selected_data

    def shuffle_dataset(self):
        all_data = []
        for fi in self.f_list:
            with open(fi, 'rb') as fr:
                while True:
                    try:
                        data = pickle.load(fr)
                        all_data.append(data)
                    except EOFError:
                        break
        random.shuffle(all_data)
        dump_file = os.path.join(self.local_path, 'shuffled_data.pkl')
        with open(dump_file, 'wb') as fw:
            for data in all_data:
                pickle.dump(data, fw)
        self.f_list = [dump_file]
        del all_data
        gc.collect()

    def remove_shuffle_file(self):
        shuffle_file = os.path.join(self.local_path, 'shuffled_data.pkl')
        if os.path.exists(shuffle_file):
            os.remove(shuffle_file)
        if self.data_path.endswith('.pkl'):
            self.f_list = [self.data_path]
        else:
            self.f_list = [os.path.join(self.data_path, fi) for fi in os.listdir(self.data_path)]

    def remove_data_by_id(self, ids):
        dump_file = os.path.join(self.local_path, 'partial_data.pkl')
        if os.path.exists(dump_file):
            temp_file = os.path.join(self.local_path, 'temp_remove.pkl')
        else:
            temp_file = dump_file
        with open(temp_file, 'wb') as fw:
            for fi in self.f_list:
                with open(fi, 'rb') as fr:
                    while True:
                        try:
                            di = pickle.load(fr)
                            d_id = di[0]
                            if d_id not in ids:
                                pickle.dump(di, fw)
                        except EOFError:
                            break
        if temp_file != dump_file:
            os.remove(dump_file)
            os.rename(temp_file, dump_file)
        self.f_list = [dump_file]

    def remove_partial_data(self):
        dump_file = os.path.join(self.local_path, 'partial_data.pkl')
        os.remove(dump_file)
        if self.data_path.endswith('.pkl'):
            self.f_list = [self.data_path]
        else:
            self.f_list = [os.path.join(self.data_path, fi) for fi in os.listdir(self.data_path)]


class RandomDataset(IterableDataset):
    def __init__(self, data_path, transforms, seed, extra_data=None):
        self.data_path = data_path
        self.extra_data = extra_data
        self.transforms = transforms
        self.seed = seed
        self.data = []
        self.load_data()
        self.rs = np.random.RandomState(seed=seed)

    def load_data(self):
        self.data.clear()
        if os.path.exists(self.data_path):
            with open(self.data_path, 'rb') as fr:
                while True:
                    try:
                        data = pickle.load(fr)
                        self.data.append(data)
                    except EOFError:
                        break
        # add support for base knowledge
        if self.extra_data is not None:
            if len(self.data) < len(self.extra_data):
                sub_extra_data = random.sample(self.extra_data, len(self.data))  # to prevent too much interference
                self.data = self.data + sub_extra_data
            else:
                self.data = self.data + self.extra_data

    def __iter__(self):
        while True:
            ind = int(self.rs.randint(low=0, high=len(self.data)))
            di = self.data[ind]
            if len(di) == 3:
                d_id, sp, lab = di
                if self.transforms is not None:
                    sp = self.transforms(sp)
                yield d_id, sp, lab
            elif len(di) == 4:
                d_id, sp, lab, logit = di
                if self.transforms is not None:
                    sp = self.transforms(sp)
                yield d_id, sp, lab, logit
            else:
                sp, lab = di
                if self.transforms is not None:
                    sp = self.transforms(sp)
                yield sp, lab


class SimpleDataset(Dataset):
    def __init__(self, data, transforms=None):
        self.data = data  # a list of samples
        self.transforms = transforms

    def set_data(self, data):
        self.data.clear()
        self.data = data

    def __getitem__(self, index):
        di = self.data[index]
        if len(di) == 3:
            d_id, sp, lab = di
            if self.transforms is not None:
                sp = self.transforms(sp)
            return d_id, sp, lab
        elif len(di) == 4:
            d_id, sp, lab, logit = di
            if self.transforms is not None:
                sp = self.transforms(sp)
            return d_id, sp, lab, logit
        else:
            sp, lab = di
            if self.transforms is not None:
                sp = self.transforms(sp)
            return sp, lab

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for di in self.data:
            if len(di) == 3:
                d_id, sp, lab = di
                if self.transforms is not None:
                    sp = self.transforms(sp)
                yield d_id, sp, lab
            elif len(di) == 4:
                d_id, sp, lab, logit = di
                if self.transforms is not None:
                    sp = self.transforms(sp)
                yield d_id, sp, lab, logit
            else:
                sp, lab = di
                if self.transforms is not None:
                    sp = self.transforms(sp)
                yield sp, lab


class DERDataset(IterableDataset):
    def __init__(self, data_path, transforms):
        self.data_path = data_path
        with open(self.data_path, 'rb') as fr:
            seen_data = pickle.load(fr)
            data = pickle.load(fr)
        self.data = data
        self.transforms = transforms

    def __iter__(self):
        for di in self.data:
            sp, lab, exp_info = di
            if self.transforms is not None:
                sp = self.transforms(sp)
            yield_data = [sp, lab]
            for i in exp_info:
                yield_data.append(i)
            yield yield_data


class NumpyDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = torch.from_numpy(data).type(torch.float)
        self.target = torch.from_numpy(target).type(torch.long)
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.data)


class MergeDataset(Dataset):
    def __init__(self, data, transforms):
        self.data = data
        self.transforms = transforms
        self.to_tensor = torchvision.transforms.ToTensor()
        self.produce_id = False

    def set_produce_id(self, produce_id):
        self.produce_id = produce_id

    def __getitem__(self, index):
        di = self.data[index]
        sp, lab = di
        aug_sp = self.transforms(sp)
        ori_sp = self.to_tensor(sp)
        if self.produce_id:
            return index, ori_sp, aug_sp, lab
        else:
            return ori_sp, aug_sp, lab

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for idx, di in enumerate(self.data):
            sp, lab = di
            aug_sp = self.transforms(sp)
            ori_sp = self.to_tensor(sp)
            if self.produce_id:
                yield idx, ori_sp, aug_sp, lab
            else:
                yield ori_sp, aug_sp, lab
