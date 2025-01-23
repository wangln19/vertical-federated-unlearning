import os
import sys
from os.path import join
import io
import json
from typing import Dict, Optional, Sequence

sys.path.append(os.pardir)

import random
import re
from collections import Counter
from copy import copy, deepcopy

import numpy as np
import pandas as pd
import pickle as pkl
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, f1_score, precision_score, roc_curve, auc, \
    average_precision_score, log_loss
from copy import deepcopy, copy
from collections import Counter

from datasets import load_dataset,Dataset
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import torch
import torchvision.transforms as transforms
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, auc, average_precision_score,
                             f1_score, log_loss, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (LabelEncoder, MinMaxScaler, OneHotEncoder,
                                   StandardScaler)
from torchvision import datasets
import torchvision.transforms as transforms
import glob

from utils.noisy_sample_functions import noisy_sample
from utils.squad_utils import *
from utils.mmlu_utils import *

from config import SEED

tp = transforms.ToTensor()
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
     ])
transform_fn = transforms.Compose([
    transforms.ToTensor()
])

from utils.basic_functions import get_class_i, get_labeled_data, fetch_data_and_label, generate_poison_data, \
    label_to_one_hot
from utils.cora_utils import *
from utils.graph_functions import load_data1, split_graph

# DATA_PATH ='./load/share_dataset/'  #'../../../share_dataset/'
DATA_PATH = '../../../share_dataset/'
IMAGE_DATA = ['mnist', 'cifar10', 'cifar100', 'cifar20', 'utkface', 'facescrub', 'places365']
TABULAR_DATA = ['breast_cancer_diagnose', 'diabetes', 'adult_income', 'criteo', 'credit', 'nursery', 'avazu']
GRAPH_DATA = ['cora']
TEXT_DATA = ['news20', 'cola_public', 'SST-2', 'STS-B', 'MRPC', 'MNLI', 'QNLI', 'QQP', 'WNLI', 'RTE', 'MMLU']


def dataset_partition_llm_new(args, dst, half_dim):
    return dataset_partition_llm(args, -1, dst, half_dim)


def dataset_partition_llm(args, index, dst, half_dim):
    '''
    dst : ( np.array(texts),np.array(label) )
    party 1 ~ k-1: Passive Party with data/label, no global model
    party k: Active Party with no data/label, but global model
    '''
    total_dim = len(dst[0])
    passive_party_num = args.k - 1

    if passive_party_num == 1:
        return dst

    if args.dataset in TEXT_DATA:
        dim_list = [0]
        for ik in range(passive_party_num - 1):
            dim_list.append(int(total_dim // (passive_party_num)) * (ik + 1))
        dim_list.append(total_dim)

        if passive_party_num == 1:
            return (dst[0], dst[1])

        elif passive_party_num == 2:
            # if index == (args.k-1): # active party has label
            # print('Passive Party Index:',index,'___',dim_list[index],':')
            active_dst = []
            for _i in range(dst[0].shape[0]):
                word_num = len(dst[0][_i]) // 2
                active_dst.append(dst[0][_i][:word_num])
            active_dst = np.array(active_dst)
            return (active_dst, dst[1])  # (dst[0][dim_list[index]:], dst[1])
            # else: # passive party does not have label
            #     if index <= (args.k-1):
            #         print('Passive Index:',index,'___',dim_list[index],':',dim_list[index+1])
            #         passive_dst = []
            #         for _i in range(dst[0].shape[0]):
            #             word_num = len(dst[0][_i]) //2
            #             passive_dst.append( dst[0][_i][word_num:] )
            #         passive_dst = np.array(passive_dst)
            #         return (passive_dst ,None) #(dst[0][dim_list[index]:dim_list[index+1]], None)
            #     else:
            #         assert index <= (args.k-1), "invalide party index"
            #         return None
        else:
            assert 1 > 2, 'partition not available'


def dataset_partition(args, index, dst, half_dim):
    if args.k == 1:
        return dst
    if args.dataset in IMAGE_DATA:
        if len(dst) == 2:  # IMAGE_DATA without attribute
            if args.k == 2:
                if index == 0:
                    return (dst[0][:, :, :half_dim, :], None)
                    # return (dst[0][:, :, half_dim:, :], None)
                elif index == 1:
                    return (dst[0][:, :, half_dim:, :], dst[1])
                    # return (dst[0][:, :, :half_dim, :], dst[1])
                else:
                    assert index <= 1, "invalide party index"
                    return None
            elif args.k == 4:
                if index == 3:
                    return (dst[0][:, :, half_dim:, half_dim:], dst[1])
                else:
                    # passive party does not have label
                    if index == 0:
                        return (dst[0][:, :, :half_dim, :half_dim], None)
                    elif index == 1:
                        return (dst[0][:, :, :half_dim, half_dim:], None)
                    elif index == 2:
                        return (dst[0][:, :, half_dim:, :half_dim], None)
                    else:
                        assert index <= 3, "invalide party index"
                        return None
            elif args.k == 1:  # Centralized Training
                return (dst[0], dst[1])
            else:
                assert (args.k == 2 or args.k == 4), "total number of parties not supported for data partitioning"
                return None
        elif len(dst) == 3:  # IMAGE_DATA with attribute
            if args.k == 2:
                if index == 0:
                    return (dst[0][:, :, :half_dim, :], None, dst[2])
                    # return (dst[0][:, :, half_dim:, :], None, None)
                elif index == 1:
                    return (dst[0][:, :, half_dim:, :], dst[1], dst[2])
                    # return (dst[0][:, :, :half_dim, :], dst[1], dst[2])
                else:
                    assert index <= 1, "invalide party index"
                    return None
            elif args.k == 4:
                if index == 3:
                    return (dst[0][:, :, half_dim:, half_dim:], dst[1], dst[2])
                else:
                    # passive party does not have label
                    if index == 0:
                        return (dst[0][:, :, :half_dim, :half_dim], None, dst[2])
                    elif index == 1:
                        return (dst[0][:, :, :half_dim, half_dim:], None, dst[2])
                    elif index == 2:
                        return (dst[0][:, :, half_dim:, :half_dim], None, dst[2])
                    else:
                        assert index <= 3, "invalide party index"
                        return None
            elif args.k == 1:  # Centralized Training
                return (dst[0], dst[1], dst[2])
            else:
                assert (args.k == 2 or args.k == 4), "total number of parties not supported for data partitioning"
                return None
    elif args.dataset in ['nuswide']:
        if args.k == 2:
            if index == 0:
                return (dst[0][0], None)  # passive party with text
            else:
                return (dst[0][1], dst[1])  # active party with image
        else:
            assert (args.k == 2), "total number of parties not supported for data partitioning"
            return None
    elif args.dataset in TABULAR_DATA or args.dataset in ['Drebin']:
        dim_list = []
        for ik in range(args.k):
            dim_list.append(int(args.model_list[str(ik)]['input_dim']))
            if len(dim_list) > 1:

                dim_list[-1] = dim_list[-1] + dim_list[-2]
                # for i in range(1, len(dim_list)):
                #     dim_list[i] = dim_list[i] + dim_list[i - 1]
        dim_list.insert(0, 0)

        if args.k == 1:  # Centralized Training
            return (dst[0], dst[1])

        if index == (args.k - 1):
            return (dst[0][:, dim_list[index]:], dst[1])
        else:
            # passive party does not have label
            if index <= (args.k - 1):
                return (dst[0][:, dim_list[index]:dim_list[index + 1]], None)
            else:
                assert index <= (args.k - 1), "invalide party index"
                return None
    elif args.dataset in TEXT_DATA:
        dim_list = []
        for ik in range(args.k):
            dim_list.append(int(args.model_list[str(ik)]['input_dim']))
            if len(dim_list) > 1:
                dim_list[-1] = dim_list[-1] + dim_list[-2]
        dim_list.insert(0, 0)

        if args.k == 1:
            return (dst[0], dst[1])

        if index == (args.k - 1):  # active party has label
            return (dst[0][:, dim_list[index]:], dst[1])
        else:  # passive party does not have label
            if index <= (args.k - 1):
                return (dst[0][:, dim_list[index]:dim_list[index + 1]], None)
            else:
                assert index <= (args.k - 1), "invalide party index"
                return None
    elif args.dataset in GRAPH_DATA:  # args.dataset == 'cora':
        assert args.k == 2, 'more than 2 party is not supported for cora'
        if index == 0:
            A_A, A_B, X_A, X_B = split_graph(args, dst[0][0], dst[0][1], split_method='com', split_ratio=0.5,
                                             with_s=True, with_f=True)
            A_A = normalize_adj(A_A)
            A_B = normalize_adj(A_B)
            # print(type(A_A),type(A_B),type(X_A),type(X_B))
            A_A = sparse_mx_to_torch_sparse_tensor(A_A).to(args.device)
            args.A_B = sparse_mx_to_torch_sparse_tensor(A_B).to(args.device)
            X_A = sparse_mx_to_torch_sparse_tensor(X_A).to(args.device)
            args.X_B = sparse_mx_to_torch_sparse_tensor(X_B).to(args.device)
            args.half_dim = [X_A.shape[1], X_B.shape[1]]
            print("cora after split", A_A.shape, A_B.shape, X_A.shape, X_B.shape)
            # print(args.half_dim)
            return ([A_A, X_A], None), args
        elif index == 1:
            return ([args.A_B, args.X_B], dst[1]), args
        else:
            assert index <= 1, 'invalid party index'

    else:
        assert args.dataset == 'mnist', f"dataset not supported {args.dataset}"
        return None
    

def index_partition(args, index):
    dim_list = []
    for ik in range(args.k):
        dim_list.append(int(args.model_list[str(ik)]['input_dim']))
        if len(dim_list) > 1:
            dim_list[-1] = dim_list[-1] + dim_list[-2]
    dim_list.insert(0, 0)

    if args.remove_specific_clients:
        #range dim_list[index]:dim_list[index + 1]
        return [i for i in range(dim_list[index], dim_list[index + 1])]

    else:  
        assert "remove_specific_clients" in args, "remove_specific_clients is not in args"
        return None


def load_dataset_per_party(args, index):
    print('load_dataset_per_party  args.need_auxiliary = ', args.need_auxiliary)
    args.classes = [None] * args.num_classes

    half_dim = -1
    args.idx_train = None
    args.idx_test = None
    if args.dataset == "cifar100":
        half_dim = 16
        train_dst = datasets.CIFAR100(DATA_PATH, download=True, train=True, transform=transform_fn)
        data, label = fetch_data_and_label(train_dst, args.num_classes)
        # train_dst = SimpleDataset(data, label)
        if args.need_auxiliary == 1:
            data, X_aux, label, y_aux = train_test_split(data, label, test_size=0.1, random_state=args.current_seed)
            X_aux = torch.tensor(X_aux)
            y_aux = torch.tensor(y_aux)
            aux_dst = (X_aux, y_aux)
        train_dst = (torch.tensor(data), label)

        test_dst = datasets.CIFAR100(DATA_PATH, download=True, train=False, transform=transform_fn)
        data, label = fetch_data_and_label(test_dst, args.num_classes)
        # test_dst = SimpleDataset(data, label)
        test_dst = (torch.tensor(data), label)
    elif args.dataset == "cifar20":
        half_dim = 16
        train_dst = datasets.CIFAR100(DATA_PATH, download=True, train=True, transform=transform_fn)
        data, label = fetch_data_and_label(train_dst, args.num_classes)
        # train_dst = SimpleDataset(data, label)
        if args.need_auxiliary == 1:
            data, X_aux, label, y_aux = train_test_split(data, label, test_size=0.1, random_state=args.current_seed)
            X_aux = torch.tensor(X_aux)
            y_aux = torch.tensor(y_aux)
            aux_dst = (X_aux, y_aux)
        train_dst = (torch.tensor(data), label)

        test_dst = datasets.CIFAR100(DATA_PATH, download=True, train=False, transform=transform_fn)
        data, label = fetch_data_and_label(test_dst, args.num_classes)
        # test_dst = SimpleDataset(data, label)
        test_dst = (torch.tensor(data), label)
    elif args.dataset == "cifar10":
        half_dim = 16
        train_dst = datasets.CIFAR10(DATA_PATH, download=True, train=True, transform=transform_fn)
        data, label = fetch_data_and_label(train_dst, args.num_classes)
        # train_dst = SimpleDataset(data, label)
        if args.need_auxiliary == 1:
            data, X_aux, label, y_aux = train_test_split(data, label, test_size=0.1, random_state=args.current_seed)
            X_aux = torch.tensor(X_aux)
            y_aux = torch.tensor(y_aux)
            aux_dst = (X_aux, y_aux)
        train_dst = (torch.tensor(data), label)

        test_dst = datasets.CIFAR10(DATA_PATH, download=True, train=False, transform=transform_fn)
        data, label = fetch_data_and_label(test_dst, args.num_classes)
        # test_dst = SimpleDataset(data, label)
        test_dst = (torch.tensor(data), label)
    elif args.dataset == "mnist":
        half_dim = 14
        train_dst = datasets.MNIST("~/.torch", download=True, train=True, transform=transform_fn)
        data, label = fetch_data_and_label(train_dst, args.num_classes)
        # train_dst = SimpleDataset(data, label)
        if args.need_auxiliary == 1:
            data, X_aux, label, y_aux = train_test_split(data, label, test_size=0.1, random_state=args.current_seed)
            X_aux = torch.tensor(X_aux)
            y_aux = torch.tensor(y_aux)
            aux_dst = (X_aux, y_aux)
            print('aux_dst:', X_aux.size(), y_aux.size())
        train_dst = (torch.tensor(data), label)

        test_dst = datasets.MNIST("~/.torch", download=True, train=False, transform=transform_fn)
        data, label = fetch_data_and_label(test_dst, args.num_classes)
        # test_dst = SimpleDataset(data, label)
        test_dst = (data, label)
    elif args.dataset == 'utkface':  # with attribute
        # 0.8 for train (all for train, but with 50% also for aux) and 0.2 for test
        half_dim = 25
        with np.load(DATA_PATH + 'UTKFace/utk_resize.npz') as f:
            data = f['imgs']
            # 'gender'=2, 'age'=11(after binning), 'race'=5
            label = f['gender' + 's']
            attribute = f['race' + 's']
            # attribute = f['age' + 's']
            # def binning_ages(a):
            #     buckets = [5, 10, 18, 25, 30, 35, 45, 55, 65, 75]
            #     for i, b in enumerate(buckets):
            #         if a <= b:
            #             return i
            #     return len(buckets)
            # attribute = [binning_ages(age) for age in attribute]
            # print(np.mean(data[:, :, :, 0]), np.mean(data[:, :, :, 1]), np.mean(data[:, :, :, 2]))
            # print(np.std(data[:, :, :, 0]), np.std(data[:, :, :, 1]), np.std(data[:, :, :, 2]))
            # MEANS = [152.13768243, 116.5061518, 99.7395918]
            # STDS = [65.71289385, 58.56545956, 57.4306078]
            MEANS = [137.10815842537994, 121.46186260277386, 112.96171130304792]
            STDS = [76.95932152349954, 74.33070450734535, 75.40728437766884]

            def channel_normalize(x):
                x = np.asarray(x, dtype=np.float32)
                x = x / 255.0
                # x[:, :, :, 0] = (x[:, :, :, 0] - MEANS[0]) / STDS[0]
                # x[:, :, :, 1] = (x[:, :, :, 1] - MEANS[1]) / STDS[1]
                # x[:, :, :, 2] = (x[:, :, :, 2] - MEANS[2]) / STDS[2]
                return x

            data = channel_normalize(data)
            label = np.asarray(label, dtype=np.int32)
            attribute = np.asarray(attribute, dtype=np.int32)
            X_train, X_test, y_train, y_test, a_train, a_test = train_test_split(data, label, attribute, train_size=0.8,
                                                                                 stratify=attribute,
                                                                                 random_state=args.current_seed)
            # [debug] in load dataset for utkface, X_aux.shape=torch.Size([9482, 50, 50, 3]), y_aux.shape=torch.Size([9482]), a_aux.shape=torch.Size([9482])
            # [debug] in load dataset for utkface, X_train.shape=torch.Size([18964, 50, 50, 3]), y_train.shape=(18964,), a_train.shape=(18964,)
            # [debug] in load dataset for utkface, X_test.shape=torch.Size([4741, 50, 50, 3]), y_test.shape=(4741,), a_test.shape=(4741,)
            # [debug] in load dataset, number of attributes for UTKFace: 5
            if args.need_auxiliary == 1:
                _, X_aux, _, y_aux, _, a_aux = train_test_split(X_train, y_train, a_train, test_size=0.5,
                                                                stratify=a_train, random_state=args.current_seed)
                # ########### counting the majority of the class ###########
                prop_counter = Counter(a_aux)
                mc = prop_counter.most_common()
                n = float(len(a_aux))
                stats = [tup[1] / n * 100 for tup in mc]
                print("Majority prop {}={:.4f}%".format(mc[0][0], stats[0]))
                print("Majority top 5={:.4f}%".format(sum(stats[:5])))
                # ########### counting the majority of the class ###########
                X_aux = torch.tensor(X_aux, dtype=torch.float32)
                y_aux = torch.tensor(y_aux, dtype=torch.long)
                a_aux = torch.tensor(a_aux, dtype=torch.long)
                # print(f"[debug] in load dataset for utkface, X_aux.shape={X_aux.shape}, y_aux.shape={y_aux.shape}, a_aux.shape={a_aux.shape}")
                aux_dst = (X_aux, y_aux, a_aux)
                # print('aux_dst:',X_aux.size(),y_aux.size())
            X_train = torch.tensor(X_train, dtype=torch.float32)
            X_test = torch.tensor(X_test, dtype=torch.float32)
            # print(f"[debug] in load dataset for utkface, X_train.shape={X_train.shape}, y_train.shape={y_train.shape}, a_train.shape={a_train.shape}")
            # print(f"[debug] in load dataset for utkface, X_test.shape={X_test.shape}, y_test.shape={y_test.shape}, a_test.shape={a_test.shape}")
            y_train = torch.tensor(y_train, dtype=torch.long)
            y_test = torch.tensor(y_test, dtype=torch.long)
            a_train = torch.tensor(a_train, dtype=torch.long)
            a_test = torch.tensor(a_test, dtype=torch.long)
            train_dst = (X_train, y_train, a_train)
            test_dst = (X_test, y_test, a_test)
            args.num_attributes = len(np.unique(a_train.numpy()))
            # print(f"[debug] in load dataset, number of attributes for UTKFace: {args.num_attributes}")
    elif args.dataset == 'facescrub':
        half_dim = 25

        def load_gender():
            i = 0
            name_gender = dict()
            for f in [DATA_PATH + 'FaceScrub/facescrub_actors.txt', DATA_PATH + 'FaceScrub/facescrub_actresses.txt']:
                with open(f) as fd:
                    fd.readline()
                    names = []
                    for line in fd.readlines():
                        components = line.split('\t')
                        assert (len(components) == 6)
                        name = components[0]  # .decode('utf8')
                        names.append(name)
                    name_gender.update(dict(zip(names, np.ones(len(names)) * i)))
                i += 1
            return name_gender

        with np.load(DATA_PATH + 'FaceScrub/Data/facescrub.npz') as f:
            data, attribute, names = [f['arr_%d' % i] for i in range(len(f.files))]

            name_gender = load_gender()
            label = [name_gender[names[i]] for i in attribute]
            label = np.asarray(label, dtype=np.int32)
            attribute = np.asarray(attribute, dtype=np.int32)
            if len(np.unique(attribute)) > 300:  # only use the most common 500 person
                id_cnt = Counter(attribute)
                attribute_selected = [tup[0] for tup in id_cnt.most_common(300)]
                indices = []
                new_attribute = []
                all_indices = np.arange(len(attribute))
                for i, face_id in enumerate(attribute_selected):
                    face_indices = all_indices[attribute == face_id]
                    new_attribute.append(np.ones_like(face_indices) * i)
                    indices.append(face_indices)
                indices = np.concatenate(indices)
                data = data[indices]
                label = label[indices]
                attribute = np.concatenate(new_attribute)
                attribute = np.asarray(attribute, dtype=np.int32)
            # print(Counter(attribute).most_common()
            X_train, X_test, y_train, y_test, a_train, a_test = train_test_split(data, label, attribute, train_size=0.8,
                                                                                 stratify=attribute,
                                                                                 random_state=args.current_seed)
            # Majority prop 0=0.5407%
            # [debug] in load dataset for FaceScrub, X_aux.shape=torch.Size([9062, 50, 50, 3]), y_aux.shape=torch.Size([9062]), a_aux.shape=torch.Size([9062])
            # [debug] in load dataset for FaceScrub, X_train.shape=torch.Size([18124, 50, 50, 3]), y_train.shape=(18124,), a_train.shape=(18124,)
            # [debug] in load dataset for FaceScrub, X_test.shape=torch.Size([4532, 50, 50, 3]), y_test.shape=(4532,), a_test.shape=(4532,)
            # [debug] in load dataset, number of attributes for FaceScrub: 300
            if args.need_auxiliary == 1:
                _, X_aux, _, y_aux, _, a_aux = train_test_split(X_train, y_train, a_train, test_size=0.5,
                                                                stratify=a_train, random_state=args.current_seed)
                # ########### counting the majority of the class ###########
                prop_counter = Counter(a_aux)
                mc = prop_counter.most_common()
                n = float(len(a_aux))
                stats = [tup[1] / n * 100 for tup in mc]
                print("Majority prop {}={:.4f}%".format(mc[0][0], stats[0]))
                print("Majority top 5={:.4f}%".format(sum(stats[:5])))
                # ########### counting the majority of the class ###########
                X_aux = torch.tensor(X_aux, dtype=torch.float32)
                y_aux = torch.tensor(y_aux, dtype=torch.long)
                a_aux = torch.tensor(a_aux, dtype=torch.long)
                print(
                    f"[debug] in load dataset for FaceScrub, X_aux.shape={X_aux.shape}, y_aux.shape={y_aux.shape}, a_aux.shape={a_aux.shape}")
                aux_dst = (X_aux, y_aux, a_aux)
                # print('aux_dst:',X_aux.size(),y_aux.size())
            X_train = torch.tensor(X_train, dtype=torch.float32)
            X_test = torch.tensor(X_test, dtype=torch.float32)
            print(
                f"[debug] in load dataset for FaceScrub, X_train.shape={X_train.shape}, y_train.shape={y_train.shape}, a_train.shape={a_train.shape}")
            print(
                f"[debug] in load dataset for FaceScrub, X_test.shape={X_test.shape}, y_test.shape={y_test.shape}, a_test.shape={a_test.shape}")
            y_train = torch.tensor(y_train, dtype=torch.long)
            y_test = torch.tensor(y_test, dtype=torch.long)
            a_train = torch.tensor(a_train, dtype=torch.long)
            a_test = torch.tensor(a_test, dtype=torch.long)
            train_dst = (X_train, y_train, a_train)
            test_dst = (X_test, y_test, a_test)
            args.num_attributes = len(np.unique(a_train.numpy()))
            print(f"[debug] in load dataset, number of attributes for FaceScrub: {args.num_attributes}")

    elif args.dataset == 'places365':
        half_dim = 64
        with np.load(DATA_PATH + 'Places365/place128.npz') as f:
            data, label, attribute = f['arr_0'], f['arr_1'], f['arr_2']
            unique_p = np.unique(attribute)
            p_to_id = dict(zip(unique_p, range(len(unique_p))))
            attribute = np.asarray([p_to_id[a] for a in attribute], dtype=np.int32)
            label = label.astype(np.int32)
            data = data / 255.0
            X_train, X_test, y_train, y_test, a_train, a_test = train_test_split(data, label, attribute, train_size=0.8,
                                                                                 stratify=attribute,
                                                                                 random_state=args.current_seed)
            # [debug] in load dataset for places365, X_aux.shape=torch.Size([29200, 128, 128, 3]), y_aux.shape=torch.Size([29200]), a_aux.shape=torch.Size([29200])
            # [debug] in load dataset for places365, X_train.shape=torch.Size([58400, 128, 128, 3]), y_train.shape=(58400,), a_train.shape=(58400,)
            # [debug] in load dataset for places365, X_test.shape=torch.Size([14600, 128, 128, 3]), y_test.shape=(14600,), a_test.shape=(14600,)
            # [debug] in load dataset, number of attributes for Places365: 365
            if args.need_auxiliary == 1:
                _, X_aux, _, y_aux, _, a_aux = train_test_split(X_train, y_train, a_train, test_size=0.5,
                                                                stratify=a_train, random_state=args.current_seed)
                # ########### counting the majority of the class ###########
                prop_counter = Counter(a_aux)
                mc = prop_counter.most_common()
                n = float(len(a_aux))
                stats = [tup[1] / n * 100 for tup in mc]
                print("Majority prop {}={:.4f}%".format(mc[0][0], stats[0]))
                print("Majority top 5={:.4f}%".format(sum(stats[:5])))
                # ########### counting the majority of the class ###########
                X_aux = torch.tensor(X_aux, dtype=torch.float32)
                y_aux = torch.tensor(y_aux, dtype=torch.long)
                a_aux = torch.tensor(a_aux, dtype=torch.long)
                # print(f"[debug] in load dataset for places365, X_aux.shape={X_aux.shape}, y_aux.shape={y_aux.shape}, a_aux.shape={a_aux.shape}")
                aux_dst = (X_aux, y_aux, a_aux)
                # print('aux_dst:',X_aux.size(),y_aux.size())
            X_train = torch.tensor(X_train, dtype=torch.float32)
            X_test = torch.tensor(X_test, dtype=torch.float32)
            # print(f"[debug] in load dataset for places365, X_train.shape={X_train.shape}, y_train.shape={y_train.shape}, a_train.shape={a_train.shape}")
            # print(f"[debug] in load dataset for places365, X_test.shape={X_test.shape}, y_test.shape={y_test.shape}, a_test.shape={a_test.shape}")
            y_train = torch.tensor(y_train, dtype=torch.long)
            y_test = torch.tensor(y_test, dtype=torch.long)
            a_train = torch.tensor(a_train, dtype=torch.long)
            a_test = torch.tensor(a_test, dtype=torch.long)
            train_dst = (X_train, y_train, a_train)
            test_dst = (X_test, y_test, a_test)
            args.num_attributes = len(np.unique(a_train.numpy()))
            # print(f"[debug] in load dataset, number of attributes for Places365: {args.num_attributes}")
    elif args.dataset == 'nuswide':
        half_dim = [1000, 634]
        if args.num_classes == 5:
            selected_labels = ['buildings', 'grass', 'animal', 'water', 'person']  # class_num = 5
        elif args.num_classes == 2:
            selected_labels = ['clouds', 'person']  # class_num = 2
            # sky 34969 light 21022
            # nature 34894 sunset 20757
            # water 31921 sea 17722
            # blue 31496 white 16938
            # clouds 26906 people 16077
            # bravo 26624 night 16057
            # landscape 23024 beach 15677
            # green 22625 architecture 15264
            # red 21983 art 14395
            # explore 21037 travel 13999

        # X_image, X_text, Y = get_labeled_data(DATA_PATH+'NUS_WIDE', selected_labels, 60, 'Train')
        X_image, X_text, Y = get_labeled_data(DATA_PATH + 'NUS_WIDE', selected_labels, 60000, 'Train')

        if args.need_auxiliary == 1:
            index_list = [_i for _i in range(0, len(X_image))]
            aux_list = random.sample(index_list, int(0.1 * len(X_image)))
            train_list = list(set(index_list) - set(aux_list))
            label = torch.squeeze(torch.tensor(np.argmax(np.array(Y), axis=1), dtype=torch.long))
            label = label_to_one_hot(label, num_classes=args.num_classes)

            X_aux = [torch.tensor(X_text[aux_list], dtype=torch.float32),
                     torch.tensor(X_image[aux_list], dtype=torch.float32)]
            y_aux = label[aux_list]  # torch.squeeze(torch.tensor(np.argmax(np.array(Y), axis=1), dtype=torch.long))
            aux_dst = (X_aux, y_aux)

            data = [torch.tensor(X_text[train_list], dtype=torch.float32),
                    torch.tensor(X_image[train_list], dtype=torch.float32)]
            label = label[train_list]
            print('nuswide dataset [aux]:', X_aux[0].shape, X_aux[1].shape, y_aux.shape)
            # print('train:',data[0].shape,data[1].shape,label.shape)
        else:
            data = [torch.tensor(X_text, dtype=torch.float32), torch.tensor(X_image, dtype=torch.float32)]
            label = torch.squeeze(torch.tensor(np.argmax(np.array(Y), axis=1), dtype=torch.long))
            label = label_to_one_hot(label, num_classes=args.num_classes)

        train_dst = (data, label)  # (torch.tensor(data),label)
        print("nuswide dataset [train]:", data[0].shape, data[1].shape, label.shape)
        # X_image, X_text, Y = get_labeled_data(DATA_PATH+'NUS_WIDE', selected_labels, 40, 'Test')
        X_image, X_text, Y = get_labeled_data(DATA_PATH + 'NUS_WIDE', selected_labels, 40000, 'Test')
        data = [torch.tensor(X_text, dtype=torch.float32), torch.tensor(X_image, dtype=torch.float32)]
        label = torch.squeeze(torch.tensor(np.argmax(np.array(Y), axis=1), dtype=torch.long))
        label = label_to_one_hot(label, num_classes=args.num_classes)
        test_dst = (data, label)
        print("nuswide dataset [test]:", data[0].shape, data[1].shape, label.shape)
    elif args.dataset in GRAPH_DATA:
        if args.dataset == 'cora':
            adj, features, idx_train, idx_val, idx_test, label = load_data1(args.dataset)
            target_nodes = idx_test
            A = np.array(adj.todense())
            X = sparse_to_tuple(features.tocoo())
            print("cora dataset before split", A.shape, type(X), X[0].shape)
            print(f"#train_sample={len(idx_train)}, #test_sample={len(idx_test)}")
            args.idx_train = torch.LongTensor(idx_train)
            args.idx_test = torch.LongTensor(idx_test)
            label = torch.LongTensor(label).to(args.device)

            # Not available for auxiliary dataset
            # if args.need_auxiliary == 1:
            #     data = [adj, features]
            #     data, X_aux, label, y_aux = train_test_split(data, label, test_size=0.1, random_state=0)
            #     X_aux = torch.tensor(X_aux)
            #     y_aux = torch.tensor(y_aux)
            #     aux_dst = (X_aux,y_aux)
            #     adj= data[0]
            #     features = data[1]

            train_dst = ([adj, features], label)
            test_dst = ([adj, features, target_nodes], label)
        half_dim = -1
    elif args.dataset in TABULAR_DATA:
        if args.dataset == 'breast_cancer_diagnose':
            half_dim = 15
            df = pd.read_csv(DATA_PATH + "BreastCancer/wdbc.data", header=0)
            X = df.iloc[:, 2:].values
            y = df.iloc[:, 1].values
            y = np.where(y == 'B', 0, 1)
            y = np.squeeze(y)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=args.current_seed)

            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.fit_transform(X_test)

        elif args.dataset == 'diabetes':
            half_dim = 4
            df = pd.read_csv(DATA_PATH + "Diabetes/diabetes.csv", header=0)
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=args.current_seed)
        elif args.dataset == 'adult_income':
            df = pd.read_csv(DATA_PATH + "Income/adult.csv", header=0)
            df = df.drop_duplicates()
            # 'age', 'workclass', 'fnlwgt', 'education', 'educational-num',
            # 'marital-status', 'occupation', 'relationship', 'race', 'gender',
            # 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
            # 'income'
            # category_columns_index = [1,3,5,6,7,8,9,13]
            # num_category_of_each_column = [9,16,7,15,6,5,2,42]
            # after one-hot encoding, the non-dummy columns are: age, fnlwgt, educational-num, capital-gain, capital-loss, hours-per-week, the first 6 columns
            category_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race',
                                'gender', 'native-country']
            for _column in category_columns:
                # Get one hot encoding of columns B
                one_hot = pd.get_dummies(df[_column], prefix=_column)
                # Drop column B as it is now encoded
                df = df.drop(_column, axis=1)
                # Join the encoded df
                df = df.join(one_hot)
            y = df['income'].values
            y = np.where(y == '<=50K', 0, 1)
            df = df.drop('income', axis=1)
            X = df.values
            # change the non-dummy columns to the last
            X = np.concatenate([X[:, 6:], X[:, :6]], axis=1)
            #Normalizing the data
            scaler = MinMaxScaler()
            X = scaler.fit_transform(X)
            # print('X shape', X.shape, X[0])
            half_dim = 6 + 9  # =15 acc=0.83
            # half_dim = 6+9+16+7+15 #=53 acc=0.77
            # half_dim = int(X.shape[1]//2) acc=0.77
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=args.current_seed)
        elif args.dataset == 'criteo':
            df = pd.read_csv(DATA_PATH + "Criteo/train.txt", sep='\t', header=None)
            df = df.sample(frac=0.02, replace=False, random_state=42)
            df.columns = ["labels"] + ["I%d" % i for i in range(1, 14)] + ["C%d" % i for i in range(14, 40)]
            print("criteo dataset loaded")
            y = df["labels"].values
            X_p = [col for col in df.columns if col.startswith('I')]
            X_a = [col for col in df.columns if col.startswith('C')]
            X_p = process_dense_feats(df, X_p)
            X_a = process_sparse_feats(df, X_a)
            print('X_p shape', X_p.shape)
            print('X_a shape', X_a.shape)
            X = pd.concat([X_a, X_p], axis=1).values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, shuffle=False)
        elif args.dataset == "credit":
            df = pd.read_csv(DATA_PATH + "tabledata/UCI_Credit_Card.csv")
            print("credit dataset loaded")

            X = df[
                [
                    "SEX",
                    "EDUCATION",
                    "MARRIAGE",
                    "AGE",
                    "LIMIT_BAL",
                    "PAY_0",
                    "PAY_2",
                    "PAY_3",
                    "PAY_4",
                    "PAY_5",
                    "PAY_6",
                    "BILL_AMT1",
                    "BILL_AMT2",
                    "BILL_AMT3",
                    "BILL_AMT4",
                    "BILL_AMT5",
                    "BILL_AMT6",
                    "PAY_AMT1",
                    "PAY_AMT2",
                    "PAY_AMT3",
                    "PAY_AMT4",
                    "PAY_AMT5",
                    "PAY_AMT6",
                ]
            ].values
            y = df["default.payment.next.month"].values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.seed, stratify=y)
            scaler = StandardScaler() # MinMaxScaler(feature_range=(0, 1))
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        elif args.dataset == "nursery":
            df = pd.read_csv(DATA_PATH + "tabledata/nursery.data", header=None)
            print("nursery dataset loaded")
            df[8] = LabelEncoder().fit_transform(df[8].values)
            X_d = df.drop(8, axis=1)
            X_a = pd.get_dummies(
                X_d[X_d.columns[: int(len(X_d.columns) / 2)]], drop_first=True, dtype=int
            )
            # print('X_a', X_a.shape)
            X_p = pd.get_dummies(
                X_d[X_d.columns[int(len(X_d.columns) / 2):]], drop_first=True, dtype=int
            )
            # print('X_p', X_p.shape)
            X = pd.concat([X_a, X_p], axis=1).values
            # print('X', X.shape)
            y = df[8].values
            scaler = StandardScaler()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.seed, stratify=y)
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        elif args.dataset == 'avazu':
            df = pd.read_csv(DATA_PATH + "avazu/train")
            df = df.sample(frac=0.02, replace=False, random_state=42)
            y = df["click"].values
            feats = process_sparse_feats(df, df.columns[2:])
            xp_idx = df.columns[-8:].tolist()
            xp_idx.insert(0, 'C1')
            xa_idx = df.columns[2:-8].tolist()
            xa_idx.remove('C1')
            X_p = feats[xp_idx]  # C14-C21
            print('X_p shape', X_p.shape)
            X_a = feats[xa_idx]
            print('X_a shape', X_a.shape)
            X = pd.concat([X_a, X_p], axis=1).values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, shuffle=False)

        if args.need_auxiliary == 1:
            X_train, X_aux, y_train, y_aux = train_test_split(X, y, test_size=0.1, random_state=args.current_seed)
            X_aux = torch.tensor(X_aux)
            y_aux = torch.tensor(y_aux)
            aux_dst = (X_aux, y_aux)
        X_train = torch.tensor(X_train)
        X_test = torch.tensor(X_test)
        y_train = torch.tensor(y_train)
        y_test = torch.tensor(y_test)
        train_dst = (X_train, y_train)
        test_dst = (X_test, y_test)
    elif args.dataset in TEXT_DATA:
        if args.dataset == 'news20':
            texts, labels, labels_index = [], {}, []
            Text_dir = DATA_PATH + 'news20/'
            for name in sorted(os.listdir(Text_dir)):
                #  every file_folder under the root_file_folder should be labels with a unique number
                labels[name] = len(labels)  #
                path = join(Text_dir, name)
                # print(sorted(os.listdir(path)))
                # for fname in sorted(os.listdir(path))[:2]:
                for fname in sorted(os.listdir(path)):
                    if fname.isdigit():  # The training set we want is all have a digit name
                        fpath = join(path, fname)
                        # print(fpath)
                        labels_index.append(labels[name])
                        # skip header
                        f = open(fpath, encoding='latin-1')
                        t = f.read()
                        texts.append(t)
                        f.close()
            # MAX_SEQUENCE_LENGTH = 1000
            # MAX_NB_WORDS = 20000
            # tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
            # tokenizer.fit_on_texts(texts)
            # sequences = tokenizer.texts_to_sequences(texts)
            # word_index = tokenizer.word_index
            # vocab_size = len(word_index) + 1
            # half_dim = int(MAX_SEQUENCE_LENGTH/2) # 500
            # X = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
            # print(labels)
            # print(len(texts))
            # print(labels_index)
            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform(texts)
            # print(X, X.shape)
            X = np.array(X.A)
            # print(X.shape) #(4702, 87748)
            y = np.array(labels_index)
            # print(y, y.shape)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.current_seed)
            # ADDED: in config: input_dim = X.shape[1]//2 need to change according to categories included
            half_dim = int(X.shape[1] // 2)  # 42491 # 43874

        if args.need_auxiliary == 1:
            X_train, X_aux, y_train, y_aux = train_test_split(X, y, test_size=0.1, random_state=args.current_seed)
            X_aux = torch.tensor(X_aux)
            y_aux = torch.tensor(y_aux)
            aux_dst = (X_aux, y_aux)

        X_train = torch.tensor(X_train)
        X_test = torch.tensor(X_test)
        y_train = torch.tensor(y_train)
        y_test = torch.tensor(y_test)
        train_dst = (X_train, y_train)
        test_dst = (X_test, y_test)
    elif args.dataset == 'Drebin':
        x_train_name, x_test_name, y_train_name, y_test_name = 'x_train', 'x_test', 'y_train.npy', 'y_test.npy'
        voc_name, features_name = 'voc.pkl', 'relevant_features.txt'
        ending = '.npz'
        x_train_name, x_test_name = x_train_name + ending, x_test_name + ending
        data_loading_fun = np.load if ending == '.npy' else sp.load_npz
        name = 'Drebin'
        DATA_PATH1 = os.path.join(DATA_PATH, name)
        X_train = data_loading_fun(os.path.join(DATA_PATH1, x_train_name))
        X_test = data_loading_fun(os.path.join(DATA_PATH1, x_test_name))
        y_train = np.load(os.path.join(DATA_PATH1, y_train_name))
        y_test = np.load(os.path.join(DATA_PATH1, y_test_name))
        # voc = pkl.load(open(os.path.join(DATA_PATH, voc_name), 'rb'))
        # relevant_features = open(os.path.join(DATA_PATH, features_name)).read().splitlines()
        # print(f"[debug] in load dataset for Drebin, X_train.shape={X_train.shape}, y_train.shape={y_train.shape}")
        # print(f"[debug] in load dataset for Drebin, X_test.shape={X_test.shape}, y_test.shape={y_test.shape}")
        # print(pd.Series(y_train).astype(int).value_counts())
        y_train = np.where(y_train == -1, 0, 1)
        y_test = np.where(y_test == -1, 0, 1)
        scaler = StandardScaler(with_mean=False)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        X_train = torch.tensor(X_train.toarray())
        X_test = torch.tensor(X_test.toarray())
        y_train = torch.tensor(y_train)
        y_test = torch.tensor(y_test)
        train_dst = (X_train, y_train)
        test_dst = (X_test, y_test)
    elif args.dataset == 'Enron':
        x_train_name, x_test_name, y_train_name, y_test_name = 'x_train', 'x_test', 'y_train.npy', 'y_test.npy'
        voc_name, features_name = 'voc.pkl', 'relevant_features.txt'
        ending = '.npz'
        x_train_name, x_test_name = x_train_name + ending, x_test_name + ending
        data_loading_fun = np.load if ending == '.npy' else sp.load_npz
        name = 'Enron'
        DATA_PATH1 = os.path.join(DATA_PATH, name)
        X_train = data_loading_fun(os.path.join(DATA_PATH1, x_train_name))
        X_test = data_loading_fun(os.path.join(DATA_PATH1, x_test_name))
        y_train = np.load(os.path.join(DATA_PATH1, y_train_name))
        y_test = np.load(os.path.join(DATA_PATH1, y_test_name))
        # voc = pkl.load(open(os.path.join(DATA_PATH, voc_name), 'rb'))
        # relevant_features = open(os.path.join(DATA_PATH, features_name)).read().splitlines()
        # print(f"[debug] in load dataset for Enron, X_train.shape={X_train.shape}, y_train.shape={y_train.shape}")
        # print(f"[debug] in load dataset for Enron, X_test.shape={X_test.shape}, y_test.shape={y_test.shape}")
        # print(pd.Series(y_train).astype(int).value_counts())
        # y_train = np.where(y_train == -1, 0, 1)
        # y_test = np.where(y_test == -1, 0, 1)
        scaler = StandardScaler(with_mean=False)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        X_train = torch.tensor(X_train.toarray())
        X_test = torch.tensor(X_test.toarray())
        y_train = torch.tensor(y_train)
        y_test = torch.tensor(y_test)
        train_dst = (X_train, y_train)
        test_dst = (X_test, y_test)
    else:
        assert args.dataset == 'mnist', "dataset not supported yet"

    if len(train_dst) == 2:
        if not args.dataset in GRAPH_DATA:
            if not args.dataset == 'nuswide':
                train_dst = (train_dst[0].to(args.device), train_dst[1].to(args.device))
                test_dst = (test_dst[0].to(args.device), test_dst[1].to(args.device))
                if args.need_auxiliary == 1:
                    aux_dst = (aux_dst[0].to(args.device), aux_dst[1].to(args.device))
            else:
                train_dst = (
                    [train_dst[0][0].to(args.device), train_dst[0][1].to(args.device)], train_dst[1].to(args.device))
                test_dst = (
                    [test_dst[0][0].to(args.device), test_dst[0][1].to(args.device)], test_dst[1].to(args.device))
                if args.need_auxiliary == 1:
                    aux_dst = (
                        [aux_dst[0][0].to(args.device), aux_dst[0][1].to(args.device)], aux_dst[1].to(args.device))
            train_dst = dataset_partition(args, index, train_dst, half_dim)
            test_dst = dataset_partition(args, index, test_dst, half_dim)
            if args.need_auxiliary == 1:
                aux_dst = dataset_partition(args, index, aux_dst, half_dim)
        else:
            train_dst, args = dataset_partition(args, index, train_dst, half_dim)
            test_dst = ([deepcopy(train_dst[0][0]), deepcopy(train_dst[0][1]), test_dst[0][2]], test_dst[1])
    elif len(train_dst) == 3:
        if not args.dataset in GRAPH_DATA:
            if not args.dataset == 'nuswide':
                train_dst = (train_dst[0].to(args.device), train_dst[1].to(args.device), train_dst[2].to(args.device))
                test_dst = (test_dst[0].to(args.device), test_dst[1].to(args.device), test_dst[2].to(args.device))
                if args.need_auxiliary == 1:
                    aux_dst = (aux_dst[0].to(args.device), aux_dst[1].to(args.device), aux_dst[2].to(args.device))
            else:
                train_dst = (
                    [train_dst[0][0].to(args.device), train_dst[0][1].to(args.device)], train_dst[1].to(args.device),
                    train_dst[2].to(args.device))
                test_dst = (
                    [test_dst[0][0].to(args.device), test_dst[0][1].to(args.device)], test_dst[1].to(args.device),
                    test_dst[2].to(args.device))
                if args.need_auxiliary == 1:
                    aux_dst = (
                        [aux_dst[0][0].to(args.device), aux_dst[0][1].to(args.device)], aux_dst[1].to(args.device),
                        aux_dst[2].to(args.device))
            train_dst = dataset_partition(args, index, train_dst, half_dim)
            test_dst = dataset_partition(args, index, test_dst, half_dim)
            if args.need_auxiliary == 1:
                aux_dst = dataset_partition(args, index, aux_dst, half_dim)
        else:
            train_dst, args = dataset_partition(args, index, train_dst, half_dim)
            test_dst = (
                [deepcopy(train_dst[0][0]), deepcopy(train_dst[0][1]), test_dst[0][2]], test_dst[1], test_dst[2])
    # important
    if args.need_auxiliary == 1:
        # print(f"[debug] aux_dst={aux_dst[0].shape},{aux_dst[1].shape if aux_dst[1] != None else aux_dst[1]}")
        # if len(aux_dst) == 3:
        #     print(f"[debug] aux_dst[2]={aux_dst[2].shape if aux_dst[2] != None else aux_dst[2]}")
        return args, half_dim, train_dst, test_dst, aux_dst
    else:
        return args, half_dim, train_dst, test_dst


def remove_features(X, percentage):
    n, d = X.shape
    num_remove = int(d * percentage)
    d = int(d * 0.25)
    remove_indices = np.random.choice(d, num_remove, replace=False)
    X[:, remove_indices] = 0
    return X, remove_indices


def remove_specific_features(X, features):
    X[:, features] = 0
    return X


def remove_samples(X, label, percentage):
    n = X.shape[0]
    num_remove = int(n * percentage)
    remove_indices = np.random.choice(n, num_remove, replace=False)
    X_deleted = X[remove_indices]
    label_deleted = label[remove_indices]
    X = np.delete(X, remove_indices, axis=0)
    label = np.delete(label, remove_indices, axis=0)
    return X, label, remove_indices, X_deleted, label_deleted


def remove_specific_samples(X, label, samples):
    X_deleted = X[samples]
    label_deleted = label[samples]
    X = np.delete(X, samples, axis=0)
    label = np.delete(label, samples, axis=0)
    return X, label, X_deleted, label_deleted


def remove_information(X, percentage):
    n, d = X.shape
    num_remove = int(n * d * percentage)
    # 2-dimensional random choice
    remove_indices = np.random.choice(n * d, num_remove, replace=False)
    # CONVERT remove_indices to 2-dimensional
    remove_indices = np.unravel_index(remove_indices, (n, d))
    X[remove_indices] = 0
    return X, remove_indices

def remove_specific_information(X, indices):
    X[indices] = 0
    return X

def load_unlearning_dataset_per_party(args, index, load_deleted=False):
    print('load_unlearning_dataset_per_party  args.need_auxiliary = ', args.need_auxiliary)
    args.classes = [None] * args.num_classes

    half_dim = -1
    args.idx_train = None
    args.idx_test = None
    if args.dataset == "cifar100":
        half_dim = 16
        train_dst = datasets.CIFAR100(DATA_PATH, download=True, train=True, transform=transform_fn)
        data, label = fetch_data_and_label(train_dst, args.num_classes)
        # train_dst = SimpleDataset(data, label)
        if args.need_auxiliary == 1:
            data, X_aux, label, y_aux = train_test_split(data, label, test_size=0.1, random_state=args.current_seed)
            X_aux = torch.tensor(X_aux)
            y_aux = torch.tensor(y_aux)
            aux_dst = (X_aux, y_aux)
        train_dst = (torch.tensor(data), label)

        test_dst = datasets.CIFAR100(DATA_PATH, download=True, train=False, transform=transform_fn)
        data, label = fetch_data_and_label(test_dst, args.num_classes)
        # test_dst = SimpleDataset(data, label)
        test_dst = (torch.tensor(data), label)
    elif args.dataset == "cifar20":
        half_dim = 16
        train_dst = datasets.CIFAR100(DATA_PATH, download=True, train=True, transform=transform_fn)
        data, label = fetch_data_and_label(train_dst, args.num_classes)
        # train_dst = SimpleDataset(data, label)
        if args.need_auxiliary == 1:
            data, X_aux, label, y_aux = train_test_split(data, label, test_size=0.1, random_state=args.current_seed)
            X_aux = torch.tensor(X_aux)
            y_aux = torch.tensor(y_aux)
            aux_dst = (X_aux, y_aux)
        train_dst = (torch.tensor(data), label)

        test_dst = datasets.CIFAR100(DATA_PATH, download=True, train=False, transform=transform_fn)
        data, label = fetch_data_and_label(test_dst, args.num_classes)
        # test_dst = SimpleDataset(data, label)
        test_dst = (torch.tensor(data), label)
    elif args.dataset == "cifar10":
        half_dim = 16
        train_dst = datasets.CIFAR10(DATA_PATH, download=True, train=True, transform=transform_fn)
        data, label = fetch_data_and_label(train_dst, args.num_classes)
        if args.remove_specific_samples:
            data, label, X_deleted, label_deleted = remove_specific_samples(data, label, args.samples_to_remove)
            remove_indices = args.samples_to_remove
        elif args.random_remove_samples_percentage:
            data, label, remove_indices, X_deleted, label_deleted = remove_samples(data, label, args.random_remove_samples_percentage)
        if load_deleted:
            assert args.remove_specific_samples or args.random_remove_samples_percentage, "Not removed any samples"
            data, label = X_deleted, label_deleted
        # train_dst = SimpleDataset(data, label)
        if args.need_auxiliary == 1:
            data, X_aux, label, y_aux = train_test_split(data, label, test_size=0.1, random_state=args.current_seed)
            X_aux = torch.tensor(X_aux)
            y_aux = torch.tensor(y_aux)
            aux_dst = (X_aux, y_aux)
        train_dst = (torch.tensor(data), label)

        test_dst = datasets.CIFAR10(DATA_PATH, download=True, train=False, transform=transform_fn)
        data, label = fetch_data_and_label(test_dst, args.num_classes)
        # test_dst = SimpleDataset(data, label)
        test_dst = (torch.tensor(data), label)
    elif args.dataset == "mnist":
        half_dim = 14
        train_dst = datasets.MNIST("~/.torch", download=True, train=True, transform=transform_fn)
        data, label = fetch_data_and_label(train_dst, args.num_classes)
        # train_dst = SimpleDataset(data, label)
        if args.need_auxiliary == 1:
            data, X_aux, label, y_aux = train_test_split(data, label, test_size=0.1, random_state=args.current_seed)
            X_aux = torch.tensor(X_aux)
            y_aux = torch.tensor(y_aux)
            aux_dst = (X_aux, y_aux)
            print('aux_dst:', X_aux.size(), y_aux.size())
        train_dst = (torch.tensor(data), label)

        test_dst = datasets.MNIST("~/.torch", download=True, train=False, transform=transform_fn)
        data, label = fetch_data_and_label(test_dst, args.num_classes)
        # test_dst = SimpleDataset(data, label)
        test_dst = (data, label)
    elif args.dataset == 'utkface':  # with attribute
        # 0.8 for train (all for train, but with 50% also for aux) and 0.2 for test
        half_dim = 25
        with np.load(DATA_PATH + 'UTKFace/utk_resize.npz') as f:
            data = f['imgs']
            # 'gender'=2, 'age'=11(after binning), 'race'=5
            label = f['gender' + 's']
            attribute = f['race' + 's']
            # attribute = f['age' + 's']
            # def binning_ages(a):
            #     buckets = [5, 10, 18, 25, 30, 35, 45, 55, 65, 75]
            #     for i, b in enumerate(buckets):
            #         if a <= b:
            #             return i
            #     return len(buckets)
            # attribute = [binning_ages(age) for age in attribute]
            # print(np.mean(data[:, :, :, 0]), np.mean(data[:, :, :, 1]), np.mean(data[:, :, :, 2]))
            # print(np.std(data[:, :, :, 0]), np.std(data[:, :, :, 1]), np.std(data[:, :, :, 2]))
            # MEANS = [152.13768243, 116.5061518, 99.7395918]
            # STDS = [65.71289385, 58.56545956, 57.4306078]
            MEANS = [137.10815842537994, 121.46186260277386, 112.96171130304792]
            STDS = [76.95932152349954, 74.33070450734535, 75.40728437766884]

            def channel_normalize(x):
                x = np.asarray(x, dtype=np.float32)
                x = x / 255.0
                # x[:, :, :, 0] = (x[:, :, :, 0] - MEANS[0]) / STDS[0]
                # x[:, :, :, 1] = (x[:, :, :, 1] - MEANS[1]) / STDS[1]
                # x[:, :, :, 2] = (x[:, :, :, 2] - MEANS[2]) / STDS[2]
                return x

            data = channel_normalize(data)
            label = np.asarray(label, dtype=np.int32)
            attribute = np.asarray(attribute, dtype=np.int32)
            X_train, X_test, y_train, y_test, a_train, a_test = train_test_split(data, label, attribute, train_size=0.8,
                                                                                 stratify=attribute,
                                                                                 random_state=args.current_seed)
            # [debug] in load dataset for utkface, X_aux.shape=torch.Size([9482, 50, 50, 3]), y_aux.shape=torch.Size([9482]), a_aux.shape=torch.Size([9482])
            # [debug] in load dataset for utkface, X_train.shape=torch.Size([18964, 50, 50, 3]), y_train.shape=(18964,), a_train.shape=(18964,)
            # [debug] in load dataset for utkface, X_test.shape=torch.Size([4741, 50, 50, 3]), y_test.shape=(4741,), a_test.shape=(4741,)
            # [debug] in load dataset, number of attributes for UTKFace: 5
            if args.need_auxiliary == 1:
                _, X_aux, _, y_aux, _, a_aux = train_test_split(X_train, y_train, a_train, test_size=0.5,
                                                                stratify=a_train, random_state=args.current_seed)
                # ########### counting the majority of the class ###########
                prop_counter = Counter(a_aux)
                mc = prop_counter.most_common()
                n = float(len(a_aux))
                stats = [tup[1] / n * 100 for tup in mc]
                print("Majority prop {}={:.4f}%".format(mc[0][0], stats[0]))
                print("Majority top 5={:.4f}%".format(sum(stats[:5])))
                # ########### counting the majority of the class ###########
                X_aux = torch.tensor(X_aux, dtype=torch.float32)
                y_aux = torch.tensor(y_aux, dtype=torch.long)
                a_aux = torch.tensor(a_aux, dtype=torch.long)
                # print(f"[debug] in load dataset for utkface, X_aux.shape={X_aux.shape}, y_aux.shape={y_aux.shape}, a_aux.shape={a_aux.shape}")
                aux_dst = (X_aux, y_aux, a_aux)
                # print('aux_dst:',X_aux.size(),y_aux.size())
            X_train = torch.tensor(X_train, dtype=torch.float32)
            X_test = torch.tensor(X_test, dtype=torch.float32)
            # print(f"[debug] in load dataset for utkface, X_train.shape={X_train.shape}, y_train.shape={y_train.shape}, a_train.shape={a_train.shape}")
            # print(f"[debug] in load dataset for utkface, X_test.shape={X_test.shape}, y_test.shape={y_test.shape}, a_test.shape={a_test.shape}")
            y_train = torch.tensor(y_train, dtype=torch.long)
            y_test = torch.tensor(y_test, dtype=torch.long)
            a_train = torch.tensor(a_train, dtype=torch.long)
            a_test = torch.tensor(a_test, dtype=torch.long)
            train_dst = (X_train, y_train, a_train)
            test_dst = (X_test, y_test, a_test)
            args.num_attributes = len(np.unique(a_train.numpy()))
            # print(f"[debug] in load dataset, number of attributes for UTKFace: {args.num_attributes}")
    elif args.dataset == 'facescrub':
        half_dim = 25

        def load_gender():
            i = 0
            name_gender = dict()
            for f in [DATA_PATH + 'FaceScrub/facescrub_actors.txt', DATA_PATH + 'FaceScrub/facescrub_actresses.txt']:
                with open(f) as fd:
                    fd.readline()
                    names = []
                    for line in fd.readlines():
                        components = line.split('\t')
                        assert (len(components) == 6)
                        name = components[0]  # .decode('utf8')
                        names.append(name)
                    name_gender.update(dict(zip(names, np.ones(len(names)) * i)))
                i += 1
            return name_gender

        with np.load(DATA_PATH + 'FaceScrub/Data/facescrub.npz') as f:
            data, attribute, names = [f['arr_%d' % i] for i in range(len(f.files))]

            name_gender = load_gender()
            label = [name_gender[names[i]] for i in attribute]
            label = np.asarray(label, dtype=np.int32)
            attribute = np.asarray(attribute, dtype=np.int32)
            if len(np.unique(attribute)) > 300:  # only use the most common 500 person
                id_cnt = Counter(attribute)
                attribute_selected = [tup[0] for tup in id_cnt.most_common(300)]
                indices = []
                new_attribute = []
                all_indices = np.arange(len(attribute))
                for i, face_id in enumerate(attribute_selected):
                    face_indices = all_indices[attribute == face_id]
                    new_attribute.append(np.ones_like(face_indices) * i)
                    indices.append(face_indices)
                indices = np.concatenate(indices)
                data = data[indices]
                label = label[indices]
                attribute = np.concatenate(new_attribute)
                attribute = np.asarray(attribute, dtype=np.int32)
            # print(Counter(attribute).most_common()
            X_train, X_test, y_train, y_test, a_train, a_test = train_test_split(data, label, attribute, train_size=0.8,
                                                                                 stratify=attribute,
                                                                                 random_state=args.current_seed)
            # Majority prop 0=0.5407%
            # [debug] in load dataset for FaceScrub, X_aux.shape=torch.Size([9062, 50, 50, 3]), y_aux.shape=torch.Size([9062]), a_aux.shape=torch.Size([9062])
            # [debug] in load dataset for FaceScrub, X_train.shape=torch.Size([18124, 50, 50, 3]), y_train.shape=(18124,), a_train.shape=(18124,)
            # [debug] in load dataset for FaceScrub, X_test.shape=torch.Size([4532, 50, 50, 3]), y_test.shape=(4532,), a_test.shape=(4532,)
            # [debug] in load dataset, number of attributes for FaceScrub: 300
            if args.need_auxiliary == 1:
                _, X_aux, _, y_aux, _, a_aux = train_test_split(X_train, y_train, a_train, test_size=0.5,
                                                                stratify=a_train, random_state=args.current_seed)
                # ########### counting the majority of the class ###########
                prop_counter = Counter(a_aux)
                mc = prop_counter.most_common()
                n = float(len(a_aux))
                stats = [tup[1] / n * 100 for tup in mc]
                print("Majority prop {}={:.4f}%".format(mc[0][0], stats[0]))
                print("Majority top 5={:.4f}%".format(sum(stats[:5])))
                # ########### counting the majority of the class ###########
                X_aux = torch.tensor(X_aux, dtype=torch.float32)
                y_aux = torch.tensor(y_aux, dtype=torch.long)
                a_aux = torch.tensor(a_aux, dtype=torch.long)
                print(
                    f"[debug] in load dataset for FaceScrub, X_aux.shape={X_aux.shape}, y_aux.shape={y_aux.shape}, a_aux.shape={a_aux.shape}")
                aux_dst = (X_aux, y_aux, a_aux)
                # print('aux_dst:',X_aux.size(),y_aux.size())
            X_train = torch.tensor(X_train, dtype=torch.float32)
            X_test = torch.tensor(X_test, dtype=torch.float32)
            print(
                f"[debug] in load dataset for FaceScrub, X_train.shape={X_train.shape}, y_train.shape={y_train.shape}, a_train.shape={a_train.shape}")
            print(
                f"[debug] in load dataset for FaceScrub, X_test.shape={X_test.shape}, y_test.shape={y_test.shape}, a_test.shape={a_test.shape}")
            y_train = torch.tensor(y_train, dtype=torch.long)
            y_test = torch.tensor(y_test, dtype=torch.long)
            a_train = torch.tensor(a_train, dtype=torch.long)
            a_test = torch.tensor(a_test, dtype=torch.long)
            train_dst = (X_train, y_train, a_train)
            test_dst = (X_test, y_test, a_test)
            args.num_attributes = len(np.unique(a_train.numpy()))
            print(f"[debug] in load dataset, number of attributes for FaceScrub: {args.num_attributes}")

    elif args.dataset == 'places365':
        half_dim = 64
        with np.load(DATA_PATH + 'Places365/place128.npz') as f:
            data, label, attribute = f['arr_0'], f['arr_1'], f['arr_2']
            unique_p = np.unique(attribute)
            p_to_id = dict(zip(unique_p, range(len(unique_p))))
            attribute = np.asarray([p_to_id[a] for a in attribute], dtype=np.int32)
            label = label.astype(np.int32)
            data = data / 255.0
            X_train, X_test, y_train, y_test, a_train, a_test = train_test_split(data, label, attribute, train_size=0.8,
                                                                                 stratify=attribute,
                                                                                 random_state=args.current_seed)
            # [debug] in load dataset for places365, X_aux.shape=torch.Size([29200, 128, 128, 3]), y_aux.shape=torch.Size([29200]), a_aux.shape=torch.Size([29200])
            # [debug] in load dataset for places365, X_train.shape=torch.Size([58400, 128, 128, 3]), y_train.shape=(58400,), a_train.shape=(58400,)
            # [debug] in load dataset for places365, X_test.shape=torch.Size([14600, 128, 128, 3]), y_test.shape=(14600,), a_test.shape=(14600,)
            # [debug] in load dataset, number of attributes for Places365: 365
            if args.need_auxiliary == 1:
                _, X_aux, _, y_aux, _, a_aux = train_test_split(X_train, y_train, a_train, test_size=0.5,
                                                                stratify=a_train, random_state=args.current_seed)
                # ########### counting the majority of the class ###########
                prop_counter = Counter(a_aux)
                mc = prop_counter.most_common()
                n = float(len(a_aux))
                stats = [tup[1] / n * 100 for tup in mc]
                print("Majority prop {}={:.4f}%".format(mc[0][0], stats[0]))
                print("Majority top 5={:.4f}%".format(sum(stats[:5])))
                # ########### counting the majority of the class ###########
                X_aux = torch.tensor(X_aux, dtype=torch.float32)
                y_aux = torch.tensor(y_aux, dtype=torch.long)
                a_aux = torch.tensor(a_aux, dtype=torch.long)
                # print(f"[debug] in load dataset for places365, X_aux.shape={X_aux.shape}, y_aux.shape={y_aux.shape}, a_aux.shape={a_aux.shape}")
                aux_dst = (X_aux, y_aux, a_aux)
                # print('aux_dst:',X_aux.size(),y_aux.size())
            X_train = torch.tensor(X_train, dtype=torch.float32)
            X_test = torch.tensor(X_test, dtype=torch.float32)
            # print(f"[debug] in load dataset for places365, X_train.shape={X_train.shape}, y_train.shape={y_train.shape}, a_train.shape={a_train.shape}")
            # print(f"[debug] in load dataset for places365, X_test.shape={X_test.shape}, y_test.shape={y_test.shape}, a_test.shape={a_test.shape}")
            y_train = torch.tensor(y_train, dtype=torch.long)
            y_test = torch.tensor(y_test, dtype=torch.long)
            a_train = torch.tensor(a_train, dtype=torch.long)
            a_test = torch.tensor(a_test, dtype=torch.long)
            train_dst = (X_train, y_train, a_train)
            test_dst = (X_test, y_test, a_test)
            args.num_attributes = len(np.unique(a_train.numpy()))
            # print(f"[debug] in load dataset, number of attributes for Places365: {args.num_attributes}")
    elif args.dataset == 'nuswide':
        half_dim = [1000, 634]
        if args.num_classes == 5:
            selected_labels = ['buildings', 'grass', 'animal', 'water', 'person']  # class_num = 5
        elif args.num_classes == 2:
            selected_labels = ['clouds', 'person']  # class_num = 2
            # sky 34969 light 21022
            # nature 34894 sunset 20757
            # water 31921 sea 17722
            # blue 31496 white 16938
            # clouds 26906 people 16077
            # bravo 26624 night 16057
            # landscape 23024 beach 15677
            # green 22625 architecture 15264
            # red 21983 art 14395
            # explore 21037 travel 13999

        # X_image, X_text, Y = get_labeled_data(DATA_PATH+'NUS_WIDE', selected_labels, 60, 'Train')
        X_image, X_text, Y = get_labeled_data(DATA_PATH + 'NUS_WIDE', selected_labels, 60000, 'Train')

        if args.need_auxiliary == 1:
            index_list = [_i for _i in range(0, len(X_image))]
            aux_list = random.sample(index_list, int(0.1 * len(X_image)))
            train_list = list(set(index_list) - set(aux_list))
            label = torch.squeeze(torch.tensor(np.argmax(np.array(Y), axis=1), dtype=torch.long))
            label = label_to_one_hot(label, num_classes=args.num_classes)

            X_aux = [torch.tensor(X_text[aux_list], dtype=torch.float32),
                     torch.tensor(X_image[aux_list], dtype=torch.float32)]
            y_aux = label[aux_list]  # torch.squeeze(torch.tensor(np.argmax(np.array(Y), axis=1), dtype=torch.long))
            aux_dst = (X_aux, y_aux)

            data = [torch.tensor(X_text[train_list], dtype=torch.float32),
                    torch.tensor(X_image[train_list], dtype=torch.float32)]
            label = label[train_list]
            print('nuswide dataset [aux]:', X_aux[0].shape, X_aux[1].shape, y_aux.shape)
            # print('train:',data[0].shape,data[1].shape,label.shape)
        else:
            data = [torch.tensor(X_text, dtype=torch.float32), torch.tensor(X_image, dtype=torch.float32)]
            label = torch.squeeze(torch.tensor(np.argmax(np.array(Y), axis=1), dtype=torch.long))
            label = label_to_one_hot(label, num_classes=args.num_classes)

        train_dst = (data, label)  # (torch.tensor(data),label)
        print("nuswide dataset [train]:", data[0].shape, data[1].shape, label.shape)
        # X_image, X_text, Y = get_labeled_data(DATA_PATH+'NUS_WIDE', selected_labels, 40, 'Test')
        X_image, X_text, Y = get_labeled_data(DATA_PATH + 'NUS_WIDE', selected_labels, 40000, 'Test')
        data = [torch.tensor(X_text, dtype=torch.float32), torch.tensor(X_image, dtype=torch.float32)]
        label = torch.squeeze(torch.tensor(np.argmax(np.array(Y), axis=1), dtype=torch.long))
        label = label_to_one_hot(label, num_classes=args.num_classes)
        test_dst = (data, label)
        print("nuswide dataset [test]:", data[0].shape, data[1].shape, label.shape)
    elif args.dataset in GRAPH_DATA:
        if args.dataset == 'cora':
            adj, features, idx_train, idx_val, idx_test, label = load_data1(args.dataset)
            target_nodes = idx_test
            A = np.array(adj.todense())
            X = sparse_to_tuple(features.tocoo())
            print("cora dataset before split", A.shape, type(X), X[0].shape)
            print(f"#train_sample={len(idx_train)}, #test_sample={len(idx_test)}")
            args.idx_train = torch.LongTensor(idx_train)
            args.idx_test = torch.LongTensor(idx_test)
            label = torch.LongTensor(label).to(args.device)

            # Not available for auxiliary dataset
            # if args.need_auxiliary == 1:
            #     data = [adj, features]
            #     data, X_aux, label, y_aux = train_test_split(data, label, test_size=0.1, random_state=0)
            #     X_aux = torch.tensor(X_aux)
            #     y_aux = torch.tensor(y_aux)
            #     aux_dst = (X_aux,y_aux)
            #     adj= data[0]
            #     features = data[1]

            train_dst = ([adj, features], label)
            test_dst = ([adj, features, target_nodes], label)
        half_dim = -1
    elif args.dataset in TABULAR_DATA:
        if args.dataset == 'breast_cancer_diagnose':
            half_dim = 15
            df = pd.read_csv(DATA_PATH + "BreastCancer/wdbc.data", header=0)
            X = df.iloc[:, 2:].values
            y = df.iloc[:, 1].values
            y = np.where(y == 'B', 0, 1)
            y = np.squeeze(y)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=args.current_seed)

            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.fit_transform(X_test)

        elif args.dataset == 'diabetes':
            half_dim = 4
            df = pd.read_csv(DATA_PATH + "Diabetes/diabetes.csv", header=0)
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=args.seed)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            X_sensitive = X_train[:, :2]
            if args.random_remove_information_percentage:
                X_sensitive, remove_indices = remove_information(X_sensitive, args.random_remove_information_percentage)
            elif args.remove_specific_information:
                X_sensitive = remove_specific_information(X_sensitive, args.information_to_remove)
                remove_indices = args.information_to_remove
            X_train = np.concatenate((X_sensitive, X_train[:, 2:]), axis=1)
        elif args.dataset == 'adult_income':
            df = pd.read_csv(DATA_PATH + "Income/adult.csv", header=0)
            df = df.drop_duplicates()
            # 'age', 'workclass', 'fnlwgt', 'education', 'educational-num',
            # 'marital-status', 'occupation', 'relationship', 'race', 'gender',
            # 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
            # 'income'
            # category_columns_index = [1,3,5,6,7,8,9,13]
            # num_category_of_each_column = [9,16,7,15,6,5,2,42]
            category_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race',
                                'gender', 'native-country']
            for _column in category_columns:
                # Get one hot encoding of columns B
                one_hot = pd.get_dummies(df[_column], prefix=_column)
                # Drop column B as it is now encoded
                df = df.drop(_column, axis=1)
                # Join the encoded df
                df = df.join(one_hot)
            y = df['income'].values
            y = np.where(y == '<=50K', 0, 1)
            df = df.drop('income', axis=1)
            X = df.values
            X = np.concatenate([X[:, 6:], X[:, :6]], axis=1)
            #Normalizing the data
            scaler = MinMaxScaler()
            X = scaler.fit_transform(X)
            # print('X shape', X.shape, X[0])
            half_dim = 6 + 9  # =15 acc=0.83
            # half_dim = 6+9+16+7+15 #=53 acc=0.77
            # half_dim = int(X.shape[1]//2) acc=0.77

            if args.remove_specific_features:
                X = remove_specific_features(X, args.features_to_remove)
                print('unlearning setting: remove specific features')
                remove_indices = args.features_to_remove

            elif args.random_remove_features_percentage:
                X, remove_indices = remove_features(X, args.random_remove_features_percentage)
                print('unlearning setting: remove random features')
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=args.current_seed)
        elif args.dataset == 'criteo':
            df = pd.read_csv(DATA_PATH + "Criteo/train.txt", sep='\t', header=None)
            df = df.sample(frac=0.02, replace=False, random_state=42)
            df.columns = ["labels"] + ["I%d" % i for i in range(1, 14)] + ["C%d" % i for i in range(14, 40)]
            print("criteo dataset loaded")
            y = df["labels"].values
            X_p = [col for col in df.columns if col.startswith('I')]
            X_a = [col for col in df.columns if col.startswith('C')]
            X_p = process_dense_feats(df, X_p)
            X_a = process_sparse_feats(df, X_a)
            print('X_p shape', X_p.shape)
            print('X_a shape', X_a.shape)
            X = pd.concat([X_a, X_p], axis=1).values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, shuffle=False)
        elif args.dataset == "credit":
            df = pd.read_csv(DATA_PATH + "tabledata/UCI_Credit_Card.csv")
            print("credit dataset loaded")

            X = df[
                [
                    "AGE",
                    "MARRIAGE",
                    "SEX",
                    "EDUCATION",
                    "LIMIT_BAL",
                    "PAY_0",
                    "PAY_2",
                    "PAY_3",
                    "PAY_4",
                    "PAY_5",
                    "PAY_6",
                    "BILL_AMT1",
                    "BILL_AMT2",
                    "BILL_AMT3",
                    "BILL_AMT4",
                    "BILL_AMT5",
                    "BILL_AMT6",
                    "PAY_AMT1",
                    "PAY_AMT2",
                    "PAY_AMT3",
                    "PAY_AMT4",
                    "PAY_AMT5",
                    "PAY_AMT6",
                ]
            ].values
            y = df["default.payment.next.month"].values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.seed, stratify=y)
            scaler = StandardScaler() # MinMaxScaler(feature_range=(0, 1))
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            X_sensitive = X_train[:, 0:2]
            if args.random_remove_information_percentage:
                X_sensitive, remove_indices = remove_information(X_sensitive, args.random_remove_information_percentage)
            elif args.remove_specific_information:
                X_sensitive = remove_specific_information(X_sensitive, args.information_to_remove)
                remove_indices = args.information_to_remove
            X_train = np.concatenate((X_sensitive, X_train[:, 2:]), axis=1)

        elif args.dataset == "nursery":
            df = pd.read_csv(DATA_PATH + "tabledata/nursery.data", header=None)
            print("nursery dataset loaded")
            df[8] = LabelEncoder().fit_transform(df[8].values)
            X_d = df.drop(8, axis=1)
            X_a = pd.get_dummies(
                X_d[X_d.columns[: int(len(X_d.columns) / 2)]], drop_first=True, dtype=int
            )
            print('X_a', X_a.shape)
            X_p = pd.get_dummies(
                X_d[X_d.columns[int(len(X_d.columns) / 2):]], drop_first=True, dtype=int
            )
            print('X_p', X_p.shape)
            X = pd.concat([X_a, X_p], axis=1).values
            print('X', X.shape)
            y = df[8].values
            scaler = StandardScaler()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.seed, stratify=y)
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            X_sensitive = X_train[:, 0:7]
            if args.random_remove_information_percentage:
                X_sensitive, remove_indices = remove_information(X_sensitive, args.random_remove_information_percentage)
            elif args.remove_specific_information:
                X_sensitive = remove_specific_information(X_sensitive, args.information_to_remove)
                remove_indices = args.information_to_remove
            X_train = np.concatenate((X_sensitive, X_train[:, 7:]), axis=1)
        elif args.dataset == 'avazu':
            df = pd.read_csv(DATA_PATH + "avazu/train")
            df = df.sample(frac=0.02, replace=False, random_state=42)
            y = df["click"].values
            feats = process_sparse_feats(df, df.columns[2:])
            xp_idx = df.columns[-8:].tolist()
            xp_idx.insert(0, 'C1')
            xa_idx = df.columns[2:-8].tolist()
            xa_idx.remove('C1')
            X_p = feats[xp_idx]  # C14-C21
            print('X_p shape', X_p.shape)
            X_a = feats[xa_idx]
            print('X_a shape', X_a.shape)
            X = pd.concat([X_a, X_p], axis=1).values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, shuffle=False)

        if args.need_auxiliary == 1:
            X_train, X_aux, y_train, y_aux = train_test_split(X, y, test_size=0.1, random_state=args.current_seed)
            X_aux = torch.tensor(X_aux)
            y_aux = torch.tensor(y_aux)
            aux_dst = (X_aux, y_aux)
        X_train = torch.tensor(X_train)
        X_test = torch.tensor(X_test)
        y_train = torch.tensor(y_train)
        y_test = torch.tensor(y_test)
        train_dst = (X_train, y_train)
        test_dst = (X_test, y_test)
    elif args.dataset in TEXT_DATA:
        if args.dataset == 'news20':
            texts, labels, labels_index = [], {}, []
            Text_dir = DATA_PATH + 'news20/'
            for name in sorted(os.listdir(Text_dir)):
                #  every file_folder under the root_file_folder should be labels with a unique number
                labels[name] = len(labels)  #
                path = join(Text_dir, name)
                # print(sorted(os.listdir(path)))
                # for fname in sorted(os.listdir(path))[:2]:
                for fname in sorted(os.listdir(path)):
                    if fname.isdigit():  # The training set we want is all have a digit name
                        fpath = join(path, fname)
                        # print(fpath)
                        labels_index.append(labels[name])
                        # skip header
                        f = open(fpath, encoding='latin-1')
                        t = f.read()
                        texts.append(t)
                        f.close()
            # MAX_SEQUENCE_LENGTH = 1000
            # MAX_NB_WORDS = 20000
            # tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
            # tokenizer.fit_on_texts(texts)
            # sequences = tokenizer.texts_to_sequences(texts)
            # word_index = tokenizer.word_index
            # vocab_size = len(word_index) + 1
            # half_dim = int(MAX_SEQUENCE_LENGTH/2) # 500
            # X = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
            # print(labels)
            # print(len(texts))
            # print(labels_index)
            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform(texts)
            # print(X, X.shape)
            X = np.array(X.A)
            # print(X, X.shape) #(4702, 87748)
            y = np.array(labels_index)
            # print(y, y.shape)

            #unlearning setting: randomly select 5% of the features to be removed

            if args.remove_specific_features:
                X = remove_specific_features(X, args.features_to_remove)
                print('unlearning setting: remove specific features')
                remove_indices = args.features_to_remove

            elif args.random_remove_features_percentage:
                X, remove_indices = remove_features(X, args.random_remove_features_percentage)
                print('unlearning setting: remove random features')       

            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.current_seed)
            # ADDED: in config: input_dim = X.shape[1]//2 need to change according to categories included
            half_dim = int(X.shape[1] // 2)  # 42491 # 43874

        if args.need_auxiliary == 1:
            X_train, X_aux, y_train, y_aux = train_test_split(X, y, test_size=0.1, random_state=args.current_seed)
            X_aux = torch.tensor(X_aux)
            y_aux = torch.tensor(y_aux)
            aux_dst = (X_aux, y_aux)

        X_train = torch.tensor(X_train)
        X_test = torch.tensor(X_test)
        y_train = torch.tensor(y_train)
        y_test = torch.tensor(y_test)
        train_dst = (X_train, y_train)
        test_dst = (X_test, y_test)

    elif args.dataset == 'Drebin':
        x_train_name, x_test_name, y_train_name, y_test_name = 'x_train', 'x_test', 'y_train.npy', 'y_test.npy'
        voc_name, features_name = 'voc.pkl', 'relevant_features.txt'
        ending = '.npz'
        x_train_name, x_test_name = x_train_name + ending, x_test_name + ending
        data_loading_fun = np.load if ending == '.npy' else sp.load_npz
        name = 'Drebin'
        DATA_PATH1 = os.path.join(DATA_PATH, name)
        X_train = data_loading_fun(os.path.join(DATA_PATH1, x_train_name))
        X_test = data_loading_fun(os.path.join(DATA_PATH1, x_test_name))
        y_train = np.load(os.path.join(DATA_PATH1, y_train_name))
        y_test = np.load(os.path.join(DATA_PATH1, y_test_name))
        # voc = pkl.load(open(os.path.join(DATA_PATH, voc_name), 'rb'))
        # relevant_features = open(os.path.join(DATA_PATH, features_name)).read().splitlines()
        # print(f"[debug] in load dataset for Drebin, X_train.shape={X_train.shape}, y_train.shape={y_train.shape}")
        # print(f"[debug] in load dataset for Drebin, X_test.shape={X_test.shape}, y_test.shape={y_test.shape}")
        # print(pd.Series(y_train).astype(int).value_counts())
        y_train = np.where(y_train == -1, 0, 1)
        y_test = np.where(y_test == -1, 0, 1)
        scaler = StandardScaler(with_mean=False)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        if args.remove_specific_features:
            X_train = remove_specific_features(X_train, args.features_to_remove)
            X_test = remove_specific_features(X_test, args.features_to_remove)
            print('unlearning setting: remove specific features')
            remove_indices = args.features_to_remove
        elif args.random_remove_features_percentage:
            X_train, remove_indices = remove_features(X_train, args.random_remove_features_percentage)
            X_test = remove_specific_features(X_test, remove_indices)
            print('unlearning setting: remove random features')
        X_train = torch.tensor(X_train.toarray())
        X_test = torch.tensor(X_test.toarray())
        y_train = torch.tensor(y_train)
        y_test = torch.tensor(y_test)
        train_dst = (X_train, y_train)
        test_dst = (X_test, y_test)
    else:
        assert args.dataset == 'mnist', "dataset not supported yet"

    if len(train_dst) == 2:
        if not args.dataset in GRAPH_DATA:
            if not args.dataset == 'nuswide':
                train_dst = (train_dst[0].to(args.device), train_dst[1].to(args.device))
                test_dst = (test_dst[0].to(args.device), test_dst[1].to(args.device))
                if args.need_auxiliary == 1:
                    aux_dst = (aux_dst[0].to(args.device), aux_dst[1].to(args.device))
            else:
                train_dst = (
                    [train_dst[0][0].to(args.device), train_dst[0][1].to(args.device)], train_dst[1].to(args.device))
                test_dst = (
                    [test_dst[0][0].to(args.device), test_dst[0][1].to(args.device)], test_dst[1].to(args.device))
                if args.need_auxiliary == 1:
                    aux_dst = (
                        [aux_dst[0][0].to(args.device), aux_dst[0][1].to(args.device)], aux_dst[1].to(args.device))
            train_dst = dataset_partition(args, index, train_dst, half_dim)
            test_dst = dataset_partition(args, index, test_dst, half_dim)
            if args.need_auxiliary == 1:
                aux_dst = dataset_partition(args, index, aux_dst, half_dim)
            # unlearning clients
            if args.remove_specific_clients:
                clients_to_remove = args.clients_to_remove
                print(f"unlearning setting: remove specific clients {clients_to_remove}")
                if index in clients_to_remove:
                    train_dst = list(train_dst)
                    test_dst = list(test_dst)
                    
                    # keep the dimensions of the dataset but the data is zeros
                    if train_dst[0] != None:
                        train_dst[0] = torch.zeros_like(train_dst[0])
                    if test_dst[0] != None:
                        test_dst[0] = torch.zeros_like(test_dst[0])
                    if train_dst[1] != None:
                        train_dst[1] = torch.zeros_like(train_dst[1])
                    if test_dst[1] != None:
                        test_dst[1] = torch.zeros_like(test_dst[1])
                    
                    train_dst = (train_dst[0], train_dst[1])
                    test_dst = (test_dst[0], test_dst[1])

                    if args.need_auxiliary == 1:
                        aux_dst = list(aux_dst)
                        aux_dst = (torch.zeros_like(aux_dst[0]), torch.zeros_like(aux_dst[1]))

        else:
            train_dst, args = dataset_partition(args, index, train_dst, half_dim)
            test_dst = ([deepcopy(train_dst[0][0]), deepcopy(train_dst[0][1]), test_dst[0][2]], test_dst[1])
    elif len(train_dst) == 3:
        if not args.dataset in GRAPH_DATA:
            if not args.dataset == 'nuswide':
                train_dst = (train_dst[0].to(args.device), train_dst[1].to(args.device), train_dst[2].to(args.device))
                test_dst = (test_dst[0].to(args.device), test_dst[1].to(args.device), test_dst[2].to(args.device))
                if args.need_auxiliary == 1:
                    aux_dst = (aux_dst[0].to(args.device), aux_dst[1].to(args.device), aux_dst[2].to(args.device))
            else:
                train_dst = (
                    [train_dst[0][0].to(args.device), train_dst[0][1].to(args.device)], train_dst[1].to(args.device),
                    train_dst[2].to(args.device))
                test_dst = (
                    [test_dst[0][0].to(args.device), test_dst[0][1].to(args.device)], test_dst[1].to(args.device),
                    test_dst[2].to(args.device))
                if args.need_auxiliary == 1:
                    aux_dst = (
                        [aux_dst[0][0].to(args.device), aux_dst[0][1].to(args.device)], aux_dst[1].to(args.device),
                        aux_dst[2].to(args.device))
            train_dst = dataset_partition(args, index, train_dst, half_dim)
            test_dst = dataset_partition(args, index, test_dst, half_dim)
            if args.need_auxiliary == 1:
                aux_dst = dataset_partition(args, index, aux_dst, half_dim)
        else:
            train_dst, args = dataset_partition(args, index, train_dst, half_dim)
            test_dst = (
                [deepcopy(train_dst[0][0]), deepcopy(train_dst[0][1]), test_dst[0][2]], test_dst[1], test_dst[2])
        
    # important
    if args.need_auxiliary == 1:
        # print(f"[debug] aux_dst={aux_dst[0].shape},{aux_dst[1].shape if aux_dst[1] != None else aux_dst[1]}")
        # if len(aux_dst) == 3:
        #     print(f"[debug] aux_dst[2]={aux_dst[2].shape if aux_dst[2] != None else aux_dst[2]}")
        return args, half_dim, train_dst, test_dst, aux_dst
    elif args.remove_specific_features or args.random_remove_features_percentage or args.remove_specific_clients or args.random_remove_samples_percentage or args.remove_specific_samples or args.remove_specific_information or args.random_remove_information_percentage:
        if args.remove_specific_clients:
            remove_indices = []
            for id in args.clients_to_remove:
                remove_indices += index_partition(args, id)
        return args, half_dim, train_dst, test_dst, remove_indices
    else:
        return args, half_dim, train_dst, test_dst


def process_dense_feats(data, feats):
    # logging.info(f"Processing feats: {feats}")
    d = data.copy()
    d = d[feats].fillna(0.0)
    for f in feats:
        d[f] = d[f].apply(lambda x: np.log(x + 1) if x > -1 else -1)
    return d


def process_sparse_feats(data, feats):
    # logging.info(f"Processing feats: {feats}")
    d = data.copy()
    d = d[feats].fillna("-1")
    for f in feats:
        label_encoder = LabelEncoder()
        d[f] = label_encoder.fit_transform(d[f])
    feature_cnt = 0
    for f in feats:
        d[f] += feature_cnt
        feature_cnt += d[f].nunique()
    return d


def prepare_poison_target_list(args):
    args.target_label = random.randint(0, args.num_classes - 1)


def load_dataset_per_party_backdoor(args, index):
    args.classes = [None] * args.num_classes

    half_dim = -1
    args.idx_train = None
    args.idx_test = None
    if args.dataset in ['mnist', 'cifar100', 'cifar20', 'cifar10']:
        # load image datasets
        if args.dataset == "cifar100":
            half_dim = 16
            train_dst = datasets.CIFAR100(DATA_PATH, download=True, train=True, transform=transform_fn)
            train_data, train_label = fetch_data_and_label(train_dst, args.num_classes)
            test_dst = datasets.CIFAR100(DATA_PATH, download=True, train=False, transform=transform_fn)
            test_data, test_label = fetch_data_and_label(test_dst, args.num_classes)
        elif args.dataset == "cifar20":
            assert args.num_classes == 20
            half_dim = 16
            train_dst = datasets.CIFAR100(DATA_PATH, download=True, train=True, transform=transform_fn)
            train_data, train_label = fetch_data_and_label(train_dst, args.num_classes)
            test_dst = datasets.CIFAR100(DATA_PATH, download=True, train=False, transform=transform_fn)
            test_data, test_label = fetch_data_and_label(test_dst, args.num_classes)
        elif args.dataset == "cifar10":
            half_dim = 16
            train_dst = datasets.CIFAR10(DATA_PATH, download=True, train=True, transform=transform_fn)
            train_data, train_label = fetch_data_and_label(train_dst, args.num_classes)
            test_dst = datasets.CIFAR10(DATA_PATH, download=True, train=False, transform=transform_fn)
            test_data, test_label = fetch_data_and_label(test_dst, args.num_classes)
        else:
            assert args.dataset == "mnist"
            half_dim = 14
            train_dst = datasets.MNIST("~/.torch", download=True, train=True, transform=transform_fn)
            train_data, train_label = fetch_data_and_label(train_dst, args.num_classes)
            test_dst = datasets.MNIST("~/.torch", download=True, train=False, transform=transform_fn)
            test_data, test_label = fetch_data_and_label(test_dst, args.num_classes)

        # poison image datasets
        if args.target_label == None:
            args.target_label = random.randint(0, args.num_classes - 1)
            args.train_poison_list = random.sample(range(len(train_dst)), int(0.01 * len(train_dst)))
            args.test_poison_list = random.sample(range(len(test_dst)), int(0.01 * len(test_dst)))
        else:
            assert args.train_poison_list != None, "[[inner error]]"
            assert args.train_target_list != None, "[[inner error]]"
            assert args.test_poison_list != None, "[[inner error]]"
            assert args.test_target_list != None, "[[inner error]]"
        print(f"party#{index} target label={args.target_label}")
        train_data, train_label, train_poison_data, train_poison_label = generate_poison_data(args, train_data,
                                                                                              train_label,
                                                                                              args.train_poison_list,
                                                                                              'train', args.k,
                                                                                              args.dataset, index)
        test_data, test_label, test_poison_data, test_poison_label = generate_poison_data(args, test_data, test_label,
                                                                                          args.test_poison_list, 'test',
                                                                                          args.k, args.dataset, index)
        if args.train_target_list == None:
            assert args.test_target_list == None
            args.train_target_list = random.sample(
                list(np.where(torch.argmax(train_label, axis=1) == args.target_label)[0]), args.num_classes)
            args.test_target_list = random.sample(
                list(np.where(torch.argmax(test_label, axis=1) == args.target_label)[0]), args.num_classes)

    elif args.dataset == 'nuswide':
        print('load backdoor data for nuswide')
        half_dim = [1000, 634]  # 634:image  1000:text
        if args.num_classes == 5:
            selected_labels = ['buildings', 'grass', 'animal', 'water', 'person']  # class_num = 5
        elif args.num_classes == 2:
            selected_labels = ['clouds', 'person']  # class_num = 2
        print('begin load')
        # X_image, X_text, Y = get_labeled_data(DATA_PATH+'NUS_WIDE', selected_labels, 6000, 'Train') # 600, too small with result in no backdoor sample
        X_image, X_text, Y = get_labeled_data(DATA_PATH + 'NUS_WIDE', selected_labels, 60000, 'Train')  # 60000
        train_data = [torch.tensor(X_text, dtype=torch.float32), torch.tensor(X_image, dtype=torch.float32)]
        train_label = torch.squeeze(torch.tensor(np.argmax(np.array(Y), axis=1), dtype=torch.long))
        print('train load over')
        # X_image, X_text, Y = get_labeled_data(DATA_PATH+'NUS_WIDE', selected_labels, 4000, 'Test') # 400, too small with result in no backdoor sample
        X_image, X_text, Y = get_labeled_data(DATA_PATH + 'NUS_WIDE', selected_labels, 40000, 'Test')  # 40000
        test_data = [torch.tensor(X_text, dtype=torch.float32), torch.tensor(X_image, dtype=torch.float32)]
        test_label = torch.squeeze(torch.tensor(np.argmax(np.array(Y), axis=1), dtype=torch.long))
        print('test load over')
        # poison image datasets
        if args.target_label == None:
            # np.array(X_text).astype('float32')
            # args.target_label = random.randint(0, args.num_classes-1)
            args.target_label = 2 if args.num_classes == 5 else random.randint(0, args.num_classes - 1)
            # print(train_data[0].shape, test_data[0].shape)
            # print("non zero train_data text", torch.nonzero(train_data[0][:,-1]))
            # print("non zero train_data text shape", torch.nonzero(train_data[0][:,-1]).shape)
            train_poison_list = torch.squeeze(torch.nonzero(train_data[0][:, -1]), dim=-1).cpu().numpy()
            test_poison_list = torch.squeeze(torch.nonzero(test_data[0][:, -1]), dim=-1).cpu().numpy()
            # print(train_poison_list[:10],test_poison_list[:10], len(train_poison_list), len(test_poison_list))
            args.train_poison_list = list(train_poison_list)
            args.test_poison_list = list(test_poison_list)
            # print(args.train_poison_list[:10],args.test_poison_list[:10], len(args.train_poison_list), len(args.test_poison_list))
        else:
            # print(args.train_poison_list, type(args.train_poison_list))
            assert args.train_poison_list != None, "[[inner error]]"
            assert args.train_target_list != None, "[[inner error]]"
            assert args.test_poison_list != None, "[[inner error]]"
            assert args.test_target_list != None, "[[inner error]]"
        print(f"party#{index} target label={args.target_label}")

        train_data, train_label, train_poison_data, train_poison_label = generate_poison_data(args, train_data,
                                                                                              train_label,
                                                                                              args.train_poison_list,
                                                                                              'train', args.k,
                                                                                              args.dataset, index)
        test_data, test_label, test_poison_data, test_poison_label = generate_poison_data(args, test_data, test_label,
                                                                                          args.test_poison_list, 'test',
                                                                                          args.k, args.dataset, index)
        if args.train_target_list == None:
            assert args.test_target_list == None
            # print('args.num_classes:',args.num_classes)
            # print('args.target_label:',args.target_label)
            # print('train_label:',train_label.size(),train_label[100:110])
            # assert 1>2
            args.train_target_list = random.sample(list(np.where(train_label == args.target_label)[0]),
                                                   args.num_classes)
            args.test_target_list = random.sample(list(np.where(test_label == args.target_label)[0]), args.num_classes)
        # transform label to onehot
        train_label = label_to_one_hot(torch.tensor(train_label), num_classes=args.num_classes)
        test_label = label_to_one_hot(torch.tensor(test_label), num_classes=args.num_classes)
        train_poison_label = label_to_one_hot(torch.tensor(train_poison_label), num_classes=args.num_classes)
        test_poison_label = label_to_one_hot(torch.tensor(test_poison_label), num_classes=args.num_classes)

    elif args.dataset in TABULAR_DATA:
        if args.dataset == 'breast_cancer_diagnose':
            half_dim = 15
            df = pd.read_csv(DATA_PATH + "BreastCancer/wdbc.data", header=0)
            X = df.iloc[:, 2:].values
            y = df.iloc[:, 1].values
            y = np.where(y == 'B', 0, 1)
            y = np.squeeze(y)
            train_data, test_data, train_label, test_label = train_test_split(X, y, test_size=0.20,
                                                                              random_state=args.current_seed)
        elif args.dataset == 'diabetes':
            half_dim = 4
            df = pd.read_csv(DATA_PATH + "Diabetes/diabetes.csv", header=0)
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            train_data, test_data, train_label, test_label = train_test_split(X, y, test_size=0.20,
                                                                              random_state=args.current_seed)
        elif args.dataset == 'adult_income':
            df = pd.read_csv(DATA_PATH + "Income/adult.csv", header=0)
            df = df.drop_duplicates()
            # 'age', 'workclass', 'fnlwgt', 'education', 'educational-num',
            # 'marital-status', 'occupation', 'relationship', 'race', 'gender',
            # 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
            # 'income'
            # category_columns_index = [1,3,5,6,7,8,9,13]
            # num_category_of_each_column = [9,16,7,15,6,5,2,42]
            category_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race',
                                'gender', 'native-country']
            for _column in category_columns:
                # Get one hot encoding of columns B
                one_hot = pd.get_dummies(df[_column], prefix=_column)
                # Drop column B as it is now encoded
                df = df.drop(_column, axis=1)
                # Join the encoded df
                df = df.join(one_hot)
            y = df['income'].values
            y = np.where(y == '<=50K', 0, 1)
            df = df.drop('income', axis=1)
            X = df.values
            half_dim = 6 + 9  # =15 acc=0.83
            # half_dim = 6+9+16+7+15 #=53 acc=0.77
            # half_dim = int(X.shape[1]//2) acc=0.77
            train_data, test_data, train_label, test_label = train_test_split(X, y, test_size=0.30,
                                                                              random_state=args.current_seed)
        elif args.dataset == 'criteo':
            df = pd.read_csv(DATA_PATH + "Criteo/criteo.csv", nrows=100000)
            print("criteo dataset loaded")
            half_dim = (df.shape[1] - 1) // 2
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            train_data, test_data, train_label, test_label = train_test_split(X, y, test_size=0.20, shuffle=False)

        train_data = torch.tensor(train_data).type(torch.float32)
        test_data = torch.tensor(test_data).type(torch.float32)
        train_label = torch.tensor(train_label)
        test_label = torch.tensor(test_label)

        # poison text datasets
        if args.target_label == None:
            args.target_label = random.randint(0, args.num_classes - 1)
            args.train_poison_list = random.sample(range(len(train_label)), int(0.01 * len(train_label)))
            args.test_poison_list = random.sample(range(len(test_label)), int(0.01 * len(test_label)))
        else:
            assert args.train_poison_list != None, "[[inner error]]"
            assert args.train_target_list != None, "[[inner error]]"
            assert args.test_poison_list != None, "[[inner error]]"
            assert args.test_target_list != None, "[[inner error]]"
        print(f"party#{index} target label={args.target_label}")

        train_data, train_label, train_poison_data, train_poison_label = generate_poison_data(args, train_data,
                                                                                              train_label,
                                                                                              args.train_poison_list,
                                                                                              'train', args.k,
                                                                                              args.dataset, index)
        test_data, test_label, test_poison_data, test_poison_label = generate_poison_data(args, test_data, test_label,
                                                                                          args.test_poison_list, 'test',
                                                                                          args.k, args.dataset, index)

        # transform label to onehot
        train_label = label_to_one_hot(torch.tensor(train_label), num_classes=args.num_classes)
        test_label = label_to_one_hot(torch.tensor(test_label), num_classes=args.num_classes)
        train_poison_label = label_to_one_hot(torch.tensor(train_poison_label), num_classes=args.num_classes)
        test_poison_label = label_to_one_hot(torch.tensor(test_poison_label), num_classes=args.num_classes)

        if args.train_target_list == None:
            assert args.test_target_list == None
            args.train_target_list = random.sample(
                list(np.where(torch.argmax(train_label, axis=1) == args.target_label)[0]), args.num_classes)
            args.test_target_list = random.sample(
                list(np.where(torch.argmax(test_label, axis=1) == args.target_label)[0]), args.num_classes)

    else:
        assert args.dataset == 'mnist', "dataset not supported yet"

    if not args.dataset == 'nuswide':
        train_dst = (train_data.to(args.device), train_label.to(args.device))
        test_dst = (test_data.to(args.device), test_label.to(args.device))
        train_poison_dst = (train_poison_data.to(args.device), train_poison_label.to(args.device))
        test_poison_dst = (test_poison_data.to(args.device), test_poison_label.to(args.device))
    else:
        train_dst = ([train_data[0].to(args.device), train_data[1].to(args.device)], train_label.to(args.device))
        test_dst = ([test_data[0].to(args.device), test_data[1].to(args.device)], test_label.to(args.device))
        train_poison_dst = ([train_poison_data[0].to(args.device), train_poison_data[1].to(args.device)],
                            train_poison_label.to(args.device))
        test_poison_dst = (
            [test_poison_data[0].to(args.device), test_poison_data[1].to(args.device)],
            test_poison_label.to(args.device))

    train_dst = dataset_partition(args, index, train_dst, half_dim)
    test_dst = dataset_partition(args, index, test_dst, half_dim)
    train_poison_dst = dataset_partition(args, index, train_poison_dst, half_dim)
    test_poison_dst = dataset_partition(args, index, test_poison_dst, half_dim)
    # important
    return args, half_dim, train_dst, test_dst, train_poison_dst, test_poison_dst, args.train_target_list, args.test_target_list


def load_dataset_per_party_noisysample(args, index):
    print(f'load_dataset_per_party_noisysample, index={index}')
    args.classes = [None] * args.num_classes

    half_dim = -1
    args.idx_train = None
    args.idx_test = None
    if args.dataset in ['mnist', 'cifar100', 'cifar20', 'cifar10']:
        # load image datasets
        if args.dataset == "cifar100":
            half_dim = 16
            train_dst = datasets.CIFAR100(DATA_PATH, download=True, train=True, transform=transform_fn)
            train_data, train_label = fetch_data_and_label(train_dst, args.num_classes)
            test_dst = datasets.CIFAR100(DATA_PATH, download=True, train=False, transform=transform_fn)
            test_data, test_label = fetch_data_and_label(test_dst, args.num_classes)
        elif args.dataset == "cifar20":
            assert args.num_classes == 20
            half_dim = 16
            train_dst = datasets.CIFAR100(DATA_PATH, download=True, train=True, transform=transform_fn)
            train_data, train_label = fetch_data_and_label(train_dst, args.num_classes)
            test_dst = datasets.CIFAR100(DATA_PATH, download=True, train=False, transform=transform_fn)
            test_data, test_label = fetch_data_and_label(test_dst, args.num_classes)
        elif args.dataset == "cifar10":
            half_dim = 16
            train_dst = datasets.CIFAR10(DATA_PATH, download=True, train=True, transform=transform_fn)
            train_data, train_label = fetch_data_and_label(train_dst, args.num_classes)
            test_dst = datasets.CIFAR10(DATA_PATH, download=True, train=False, transform=transform_fn)
            test_data, test_label = fetch_data_and_label(test_dst, args.num_classes)
        else:
            assert args.dataset == "mnist"
            half_dim = 14
            train_dst = datasets.MNIST("~/.torch", download=True, train=True, transform=transform_fn)
            train_data, train_label = fetch_data_and_label(train_dst, args.num_classes)
            test_dst = datasets.MNIST("~/.torch", download=True, train=False, transform=transform_fn)
            test_data, test_label = fetch_data_and_label(test_dst, args.num_classes)

        # poison image datasets
        assert 'noise_lambda' in args.attack_configs, 'need parameter: noise_lambda'
        assert 'noise_rate' in args.attack_configs, 'need parameter: noise_rate'
        assert 'party' in args.attack_configs, 'need parameter: party'
        noise_rate = args.attack_configs['noise_rate'] if ('noise_rate' in args.attack_configs) else 0.01
        scale = args.attack_configs['noise_lambda'] if ('noise_lambda' in args.attack_configs) else 2.0
        if not index in args.attack_configs['party']:
            scale = 0.0

        if args.train_poison_list == None:
            assert args.test_poison_list == None, "[[inner error]]"
            args.train_poison_list = random.sample(range(len(train_dst)), int(noise_rate * len(train_dst)))
            args.test_poison_list = random.sample(range(len(test_dst)), int(noise_rate * len(test_dst)))
            print(len(train_dst), len(test_dst), len(args.train_poison_list), len(args.test_poison_list))
        else:
            assert args.test_poison_list != None, "[[inner error]]"

        train_data, train_label, train_poison_data, train_poison_label = generate_poison_data(args, train_data,
                                                                                              train_label,
                                                                                              args.train_poison_list,
                                                                                              'train', args.k,
                                                                                              args.dataset, index)
        test_data, test_label, test_poison_data, test_poison_label = generate_poison_data(args, test_data, test_label,
                                                                                          args.test_poison_list, 'test',
                                                                                          args.k, args.dataset, index)

    elif args.dataset == 'nuswide':
        print('load backdoor data for nuswide')
        half_dim = [1000, 634]  # 634:image  1000:text
        if args.num_classes == 5:
            selected_labels = ['buildings', 'grass', 'animal', 'water', 'person']  # class_num = 5
        elif args.num_classes == 2:
            selected_labels = ['clouds', 'person']  # class_num = 2
        print('begin load')
        # X_image, X_text, Y = get_labeled_data(DATA_PATH+'NUS_WIDE', selected_labels, 6000, 'Train') # 600, too small with result in no backdoor sample
        X_image, X_text, Y = get_labeled_data(DATA_PATH + 'NUS_WIDE', selected_labels, 60000, 'Train')  # 60000
        train_data = [torch.tensor(X_text, dtype=torch.float32), torch.tensor(X_image, dtype=torch.float32)]
        train_label = torch.squeeze(torch.tensor(np.argmax(np.array(Y), axis=1), dtype=torch.long))
        print('train load over')
        # X_image, X_text, Y = get_labeled_data(DATA_PATH+'NUS_WIDE', selected_labels, 4000, 'Test') # 400, too small with result in no backdoor sample
        X_image, X_text, Y = get_labeled_data(DATA_PATH + 'NUS_WIDE', selected_labels, 40000, 'Test')  # 40000
        test_data = [torch.tensor(X_text, dtype=torch.float32), torch.tensor(X_image, dtype=torch.float32)]
        test_label = torch.squeeze(torch.tensor(np.argmax(np.array(Y), axis=1), dtype=torch.long))
        print('test load over')

        # poison image datasets
        assert 'noise_lambda' in args.attack_configs, 'need parameter: noise_lambda'
        assert 'noise_rate' in args.attack_configs, 'need parameter: noise_rate'
        assert 'party' in args.attack_configs, 'need parameter: party'
        noise_rate = args.attack_configs['noise_rate'] if ('noise_rate' in args.attack_configs) else 0.01
        scale = args.attack_configs['noise_lambda'] if ('noise_lambda' in args.attack_configs) else 2.0
        if not index in args.attack_configs['party']:
            scale = 0.0

        if args.train_poison_list == None:
            assert args.test_poison_list == None, "[[inner error]]"
            args.train_poison_list = random.sample(range(len(train_data[0])), int(noise_rate * len(train_data[0])))
            args.test_poison_list = random.sample(range(len(test_data[0])), int(noise_rate * len(test_data[0])))
        else:
            assert args.test_poison_list != None, "[[inner error]]"

        train_data, train_label, train_poison_data, train_poison_label = generate_poison_data(args, train_data,
                                                                                              train_label,
                                                                                              args.train_poison_list,
                                                                                              'train', args.k,
                                                                                              args.dataset, index)
        test_data, test_label, test_poison_data, test_poison_label = generate_poison_data(args, test_data, test_label,
                                                                                          args.test_poison_list, 'test',
                                                                                          args.k, args.dataset, index)

        # transform label to onehot
        train_label = label_to_one_hot(torch.tensor(train_label), num_classes=args.num_classes)
        test_label = label_to_one_hot(torch.tensor(test_label), num_classes=args.num_classes)
        train_poison_label = label_to_one_hot(torch.tensor(train_poison_label), num_classes=args.num_classes)
        test_poison_label = label_to_one_hot(torch.tensor(test_poison_label), num_classes=args.num_classes)

    elif args.dataset in TABULAR_DATA:
        if args.dataset == 'breast_cancer_diagnose':
            half_dim = 15
            df = pd.read_csv(DATA_PATH + "BreastCancer/wdbc.data", header=0)
            X = df.iloc[:, 2:].values
            y = df.iloc[:, 1].values
            y = np.where(y == 'B', 0, 1)
            y = np.squeeze(y)
            train_data, test_data, train_label, test_label = train_test_split(X, y, test_size=0.20,
                                                                              random_state=args.current_seed)
        elif args.dataset == 'diabetes':
            half_dim = 4
            df = pd.read_csv(DATA_PATH + "Diabetes/diabetes.csv", header=0)
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            train_data, test_data, train_label, test_label = train_test_split(X, y, test_size=0.20,
                                                                              random_state=args.current_seed)
        elif args.dataset == 'adult_income':
            df = pd.read_csv(DATA_PATH + "Income/adult.csv", header=0)
            df = df.drop_duplicates()
            # 'age', 'workclass', 'fnlwgt', 'education', 'educational-num',
            # 'marital-status', 'occupation', 'relationship', 'race', 'gender',
            # 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
            # 'income'
            # category_columns_index = [1,3,5,6,7,8,9,13]
            # num_category_of_each_column = [9,16,7,15,6,5,2,42]
            category_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race',
                                'gender', 'native-country']
            for _column in category_columns:
                # Get one hot encoding of columns B
                one_hot = pd.get_dummies(df[_column], prefix=_column)
                # Drop column B as it is now encoded
                df = df.drop(_column, axis=1)
                # Join the encoded df
                df = df.join(one_hot)
            y = df['income'].values
            y = np.where(y == '<=50K', 0, 1)
            df = df.drop('income', axis=1)
            X = df.values
            half_dim = 6 + 9  # =15 acc=0.83
            # half_dim = 6+9+16+7+15 #=53 acc=0.77
            # half_dim = int(X.shape[1]//2) acc=0.77
            train_data, test_data, train_label, test_label = train_test_split(X, y, test_size=0.30,
                                                                              random_state=args.current_seed)
        elif args.dataset == 'criteo':
            df = pd.read_csv(DATA_PATH + "Criteo/criteo.csv", nrows=100000)
            print("criteo dataset loaded")
            half_dim = (df.shape[1] - 1) // 2
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            train_data, test_data, train_label, test_label = train_test_split(X, y, test_size=0.20, shuffle=False)

        train_data = torch.tensor(train_data).type(torch.float32)
        test_data = torch.tensor(test_data).type(torch.float32)
        train_label = torch.tensor(train_label)
        test_label = torch.tensor(test_label)

        # poison image datasets
        assert 'noise_lambda' in args.attack_configs, 'need parameter: noise_lambda'
        assert 'noise_rate' in args.attack_configs, 'need parameter: noise_rate'
        assert 'party' in args.attack_configs, 'need parameter: party'
        noise_rate = args.attack_configs['noise_rate'] if ('noise_rate' in args.attack_configs) else 0.1
        scale = args.attack_configs['noise_lambda']
        if not index in args.attack_configs['party']:
            scale = 0.0

        if args.train_poison_list == None:
            assert args.test_poison_list == None, "[[inner error]]"
            args.train_poison_list = random.sample(range(len(train_dst)), int(noise_rate * len(train_dst)))
            args.test_poison_list = random.sample(range(len(test_dst)), int(noise_rate * len(test_dst)))
        else:
            assert args.test_poison_list != None, "[[inner error]]"

        train_data, train_label, train_poison_data, train_poison_label = generate_poison_data(args, train_data,
                                                                                              train_label,
                                                                                              args.train_poison_list,
                                                                                              'train', args.k,
                                                                                              args.dataset, index)
        test_data, test_label, test_poison_data, test_poison_label = generate_poison_data(args, test_data, test_label,
                                                                                          args.test_poison_list, 'test',
                                                                                          args.k, args.dataset, index)

        # transform label to onehot
        train_label = label_to_one_hot(torch.tensor(train_label), num_classes=args.num_classes)
        test_label = label_to_one_hot(torch.tensor(test_label), num_classes=args.num_classes)
        train_poison_label = label_to_one_hot(torch.tensor(train_poison_label), num_classes=args.num_classes)
        test_poison_label = label_to_one_hot(torch.tensor(test_poison_label), num_classes=args.num_classes)

    else:
        assert args.dataset == 'mnist', "dataset not supported yet"

    if not args.dataset == 'nuswide':
        train_dst = (train_data.to(args.device), train_label.to(args.device))
        test_dst = (test_data.to(args.device), test_label.to(args.device))
        train_poison_dst = (train_poison_data.to(args.device), train_poison_label.to(args.device))
        test_poison_dst = (test_poison_data.to(args.device), test_poison_label.to(args.device))
    else:
        train_dst = ([train_data[0].to(args.device), train_data[1].to(args.device)], train_label.to(args.device))
        test_dst = ([test_data[0].to(args.device), test_data[1].to(args.device)], test_label.to(args.device))
        train_poison_dst = ([train_poison_data[0].to(args.device), train_poison_data[1].to(args.device)],
                            train_poison_label.to(args.device))
        test_poison_dst = (
            [test_poison_data[0].to(args.device), test_poison_data[1].to(args.device)],
            test_poison_label.to(args.device))

    train_dst = dataset_partition(args, index, train_dst, half_dim)
    test_dst = dataset_partition(args, index, test_dst, half_dim)
    train_poison_dst = dataset_partition(args, index, train_poison_dst, half_dim)
    test_poison_dst = dataset_partition(args, index, test_poison_dst, half_dim)
    # important
    return args, half_dim, train_dst, test_dst, train_poison_dst, test_poison_dst


def load_mnli(file, header=True, multi_snli=False, is_train=True):
    rows = []
    cnt = 0
    with open(file, encoding="utf8") as f:
        for line in f:
            if header:
                header = False
                continue
            blocks = line.strip().split("\t")
            assert len(blocks) > 9
            if blocks[-1] == "-":
                continue
            lab = "contradiction"
            if is_train:
                lab = blocks[-1]
            if lab is None:
                import pdb

                pdb.set_trace()
            sample = {
                "uid": blocks[0],
                "premise": blocks[8],
                "hypothesis": blocks[9],
                "label": lab,
            }
            rows.append(sample)
            cnt += 1
    return rows


# def format_subject(subject):
#     l = subject.split("_")
#     s = ""
#     for entry in l:
#         s += " " + entry
#     return s


# def format_example(df, idx, choices, include_answer=True):
#     prompt = df.iloc[idx, 0]
#     k = len(choices)
#     for j in range(k):
#         prompt += "\n{}. {}".format(choices[j], df.iloc[idx][str(choices[j])])
#     prompt += "\nAnswer:"
#     if include_answer:
#         prompt += " {}\n\n".format(df.iloc[idx]['answer'])  # df.iloc[idx, k + 1]
#     answer = df.iloc[idx]['answer']
#     return prompt, answer


# def gen_prompt(train_df, choices, subject, k=-1):
#     prompt_list = []
#     if k == -1:
#         k = train_df.shape[0]

#     _num = 0

#     for i in range(len(train_df)):
#         if train_df.iloc[i]['subject'] == subject:
#             prompt, answer = format_example(train_df, i, choices)
#             prompt_list.append(prompt)
#             _num += 1
#             if _num >= k:
#                 break

#     return prompt_list


def load_dataset_per_party_llm(args, index):
    print('load_dataset_per_party_llm  args.need_auxiliary = ', args.need_auxiliary)
    args.classes = [None] * args.num_classes

    half_dim = -1
    args.idx_train = None
    args.idx_test = None

    if args.dataset == 'news20':
        texts, labels, labels_index = [], {}, []
        Text_dir = DATA_PATH + 'news20/'
        for name in sorted(os.listdir(Text_dir)[:2]):
            #  every file_folder under the root_file_folder should be labels with a unique number
            labels[name] = len(labels)  #
            path = join(Text_dir, name)
            for fname in sorted(os.listdir(path)):
                if fname.isdigit():  # The training set we want is all have a digit name
                    fpath = join(path, fname)
                    # labels_index.append(labels[name])
                    # skip header
                    f = open(fpath, encoding='latin-1')
                    t = f.read()

                    # tokenized_text = args.tokenizer(t,padding='max_length',
                    #    max_length = args.max_sequence,
                    #    truncation=True,
                    #    return_tensors="pt")

                    texts.append(t)

                    # ids = args.tokenizer(t, truncation=True, max_length=args.max_sequence, padding='max_length',return_tensors="pt")
                    # texts.append( torch.tensor(ids['input_ids']).squeeze() )

                    # # input_ids.append( tokenized_text['input_ids'] )
                    # # token_type_ids.append( tokenized_text['token_type_ids'] )
                    # # attention_mask.append( tokenized_text['attention_mask'] )

                    labels_index.append(labels[name])
                    f.close()

        # texts=[aa.tolist() for aa in texts]#列表中元素由tensor变成列表。
        # X = torch.tensor( texts)
        X = np.array(texts)
        y = np.array(labels_index)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.current_seed)
        # token_type_ids_train, token_type_ids_test, y_train, y_test = train_test_split(token_type_ids, y, test_size=0.2, random_state=args.current_seed)
        # attention_mask_train, attention_mask_test, y_train, y_test = train_test_split(attention_mask, y, test_size=0.2, random_state=args.current_seed)

        print('X:', X_train.shape, X_test.shape)  # (1600,3) (400,3)
        print('y:', y_train.shape, y_test.shape)  # (1600,) (400,)

        # if args.need_auxiliary == 1:
        #     X_train, X_aux, y_train, y_aux = train_test_split(X, y, test_size=0.1, random_state=args.current_seed)
        #     X_aux = torch.tensor(X_aux)
        #     y_aux = torch.tensor(y_aux)
        #     aux_dst = (X_aux,y_aux)

        # X_train = torch.tensor(X_train)
        # X_test = torch.tensor(X_test)
        # y_train = torch.tensor(y_train)
        # y_test = torch.tensor(y_test)

        train_dst = (X_train, y_train)
        test_dst = (X_test, y_test)

    elif args.dataset == 'CoLA':
        train_set_file, test_set_file = get_dataset_path(args.model_list[str(index)])
        if train_set_file is None or test_set_file is None:
            train_set_file = DATA_PATH + 'CoLA/raw/in_domain_train.tsv'
            test_set_file = DATA_PATH + 'CoLA/raw/in_domain_dev.tsv'

        df = pd.read_csv(train_set_file, delimiter='\t', header=None,
                         names=['sentence_source', 'label', 'label_notes', 'sentence'])  # [:100]
        sentences = df.sentence.values
        labels = df.label.values
        X_train = np.array(sentences)
        y_train = np.array(labels)

        df = pd.read_csv(test_set_file, delimiter='\t', header=None,
                         names=['sentence_source', 'label', 'label_notes', 'sentence'])  # [:10]
        sentences = df.sentence.values
        labels = df.label.values
        X_test = np.array(sentences)
        y_test = np.array(labels)

        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.current_seed)

        print(type(X_train), X_train.shape, X_test.shape)  # (6840,512) (1711,512)
        print(type(y_train), y_train.shape, y_test.shape)  # (6840,1) (1711,1)
        train_dst = (X_train, y_train)
        test_dst = (X_test, y_test)

    elif args.dataset == 'yelp-polarity':
        train_set_file = DATA_PATH + 'Yelp/yelp_review_full_csv/train.csv'
        test_set_file = DATA_PATH + 'Yelp/yelp_review_full_csv/test.csv'

        df = pd.read_csv(train_set_file, delimiter=',', header=None,
                         names=['label', 'sentence'])  # [:5000]

        scalar = np.array([-1])
        sentences = df.sentence.values
        labels = df.label.values
        X_train = np.array(sentences)
        y_train = np.array(labels) + scalar

        df = pd.read_csv(test_set_file, delimiter=',', header=None,
                         names=['label', 'sentence'])  # [:500]

        sentences = df.sentence.values
        labels = df.label.values
        X_test = np.array(sentences)
        y_test = np.array(labels) + scalar

        train_dst = (X_train, y_train)
        test_dst = (X_test, y_test)

        print('X:', type(X_train), X_train.shape, X_test.shape)
        print('y:', type(y_train), y_train.shape, y_test.shape)

    elif args.dataset == "emotion":
        X_train = []
        y_train = []
        X_test = []
        y_test = []
        text_path = DATA_PATH + 'emotion/data/data.jsonl'
        with open(text_path, "r") as file:
            for line in file:
                data = json.loads(line)
                X_train.append(data["text"])
                y_train.append(data["label"])

        text_path = DATA_PATH + 'emotion/data/train.jsonl'
        with open(text_path, "r") as file:
            for line in file:
                data = json.loads(line)
                X_train.append(data["text"])
                y_train.append(data["label"])

        text_path = DATA_PATH + 'emotion/data/test.jsonl'
        with open(text_path, "r") as file:
            for line in file:
                data = json.loads(line)
                X_test.append(data["text"])
                y_test.append(data["label"])

        text_path = DATA_PATH + 'emotion/data/validation.jsonl'
        with open(text_path, "r") as file:
            for line in file:
                data = json.loads(line)
                X_test.append(data["text"])
                y_test.append(data["label"])

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        X_test = np.array(X_test)
        y_test = np.array(y_test)

        train_dst = (X_train, y_train)
        test_dst = (X_test, y_test)


    elif args.dataset == 'SST-2':
        # task_prompt = {
        # 'imdb': """Analyze the following movie review and determine if the sentiment is: positive or negative. Return answer in single word as either positive or negative: {}""",
        # "yelp": """Analyze the following restaurant review and determine if the sentiment is: positive or negative. Return answer in single word as either positive or negative: {}""",
        # "SST-2": """Analyze the following sentence and determine if the sentiment is: positive or negative.\n{}\nThe awnser is:""",
        #     }
        train_set_file, test_set_file = get_dataset_path(args.model_list[str(index)])
        if train_set_file is None or test_set_file is None:
            train_set_file = DATA_PATH + 'SST-2/train.tsv'
            test_set_file = DATA_PATH + 'SST-2/dev.tsv'
        df = pd.read_csv(train_set_file, delimiter='\t', names=['label', 'sentence'])  # names=[  'sentence','label']
        sentences = df.sentence.values[1:]
        labels = df.label.values[1:]

        # if args.model_architect == 'CLM':
        #     instructions = task_prompt[args.dataset]
        #     for i in range(len(sentences)):
        #         sentences[i] = instructions.format(sentences[i])

        X_train = np.array(sentences)
        y_train = np.array([int(_label) for _label in labels])

        df = pd.read_csv(test_set_file, delimiter='\t', names=['label', 'sentence'])  # names=[  'sentence','label']
        sentences = df.sentence.values[1:]
        labels = df.label.values[1:]

        # if args.model_architect == 'CLM':
        #     instructions = task_prompt[args.dataset]
        #     for i in range(len(sentences)):
        #         sentences[i] = instructions.format(sentences[i])

        X_test = np.array(sentences)
        y_test = np.array([int(_label) for _label in labels])
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.current_seed)

        print(type(X_train), X_train.shape, X_test.shape)  #
        print(type(y_train), y_train.shape, y_test.shape)  #

        train_dst = (X_train, y_train)
        test_dst = (X_test, y_test)

    elif args.dataset == 'STS-B':
        text_path = DATA_PATH + 'STS-B/train.tsv'
        df = pd.read_csv(text_path, sep='\t', on_bad_lines="skip")
        df = df.dropna()
        sentence_pairs = np.array(list(zip(df.sentence1.values, df.sentence2.values)))
        labels = df.score.values

        X_train = np.array(sentence_pairs)
        y_train = np.array(labels)

        text_path = DATA_PATH + 'STS-B/dev.tsv'
        df = pd.read_csv(text_path, sep='\t', on_bad_lines="skip")  # , error_bad_lines=False)
        df = df.dropna()
        sentence_pairs = np.array(list(zip(df.sentence1.values, df.sentence2.values)))
        labels = df.score.values

        X_test = np.array(sentence_pairs)
        y_test = np.array(labels)
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.current_seed)

        print(type(X_train), X_train.shape, X_test.shape)  #
        print(type(y_train), y_train.shape, y_test.shape)  #

        train_dst = (X_train, y_train)
        test_dst = (X_test, y_test)

    elif args.dataset == 'MRPC':
        text_path = DATA_PATH + 'MRPC/train.tsv'
        df = pd.read_csv(text_path, sep='\t',
                         on_bad_lines='skip')  # sep='\t',error_bad_lines=False)# names=[  'sentence','label']
        df.columns = ['Quality', 'id1', 'id2', 'sentence1', 'sentence2']
        sentence_pairs = np.array(list(zip(df.sentence1.values, df.sentence2.values)))
        labels = df.Quality.values

        X_train = np.array(sentence_pairs)
        y_train = np.array(labels)

        text_path = DATA_PATH + 'MRPC/dev.tsv'
        df = pd.read_csv(text_path, sep='\t', on_bad_lines='skip')  # ,error_bad_lines=False)
        df.columns = ['Quality', 'id1', 'id2', 'sentence1', 'sentence2']
        sentence_pairs = np.array(list(zip(df.sentence1.values, df.sentence2.values)))
        labels = df.Quality.values

        X_test = np.array(sentence_pairs)
        y_test = np.array(labels)
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.current_seed)

        print(type(X_train), X_train.shape, X_test.shape)  #
        print(type(y_train), y_train.shape, y_test.shape)  #

        train_dst = (X_train, y_train)
        test_dst = (X_test, y_test)

    elif args.dataset == 'MNLI':
        # label_dict={
        #     'entailment': 1,
        #     'neutral': 2,
        #     'contradiction': 0
        # }
        label_dict = {
            'entailment': 0,
            'neutral': 1,
            'contradiction': 2
        }

        text_path = DATA_PATH + 'MNLI/train.tsv'
        data = load_mnli(text_path)
        print("Loaded {} MNLI train samples".format(len(data)))
        # text_path = DATA_PATH + 'MNLI/train.tsv'
        # df = pd.read_csv(text_path , sep='\t',error_bad_lines=False)[:5000]# names=[  'sentence','label']
        sentence_pairs = [[_data['premise'], _data['hypothesis']] for _data in data]
        labels = [_data['label'] for _data in data]
        sentence_pairs = np.array(sentence_pairs)
        labels = [label_dict[_label] for _label in labels]

        X_train = np.array(sentence_pairs)
        y_train = np.array(labels)

        text_path = DATA_PATH + 'MNLI/dev_matched.tsv'
        data = load_mnli(text_path)
        print("Loaded {} MNLI dev samples".format(len(data)))
        # text_path = DATA_PATH + 'MNLI/train.tsv'
        # df = pd.read_csv(text_path , sep='\t',error_bad_lines=False)[:5000]# names=[  'sentence','label']
        sentence_pairs = [[_data['premise'], _data['hypothesis']] for _data in data]
        labels = [_data['label'] for _data in data]
        sentence_pairs = np.array(sentence_pairs)
        labels = [label_dict[_label] for _label in labels]

        X_test = np.array(sentence_pairs)
        y_test = np.array(labels)
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.current_seed)

        print(type(X_train), X_train.shape, X_test.shape)  #
        print(type(y_train), y_train.shape, y_test.shape, type(y_train[0]))  #

        train_dst = (X_train, y_train)
        test_dst = (X_test, y_test)

    elif args.dataset == 'QNLI':
        label_dict = {
            'entailment': 0,
            'not_entailment': 1
        }
        text_path = DATA_PATH + 'QNLI/train.tsv'
        df = pd.read_csv(text_path, sep='\t', error_bad_lines=False)
        sentence_pairs = np.array(list(zip(df.question.values, df.sentence.values)))
        labels = df.label.values
        labels = [label_dict[_label] for _label in labels]

        X_train = np.array(sentence_pairs)
        y_train = np.array(labels)

        text_path = DATA_PATH + 'QNLI/dev.tsv'
        df = pd.read_csv(text_path, sep='\t', error_bad_lines=False)
        sentence_pairs = np.array(list(zip(df.question.values, df.sentence.values)))
        labels = df.label.values
        labels = [label_dict[_label] for _label in labels]

        X_test = np.array(sentence_pairs)
        y_test = np.array(labels)
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.current_seed)

        print(type(X_train), X_train.shape, X_test.shape)  #
        print(type(y_train), y_train.shape, y_test.shape)  #

        train_dst = (X_train, y_train)
        test_dst = (X_test, y_test)

    elif args.dataset == 'QQP':
        text_path = DATA_PATH + 'QQP/train.tsv'
        df = pd.read_csv(text_path, sep='\t', error_bad_lines=False)
        sentence_pairs = np.array(list(zip(df.question1.values, df.question2.values)))
        labels = df.is_duplicate.values

        X_train = np.array(sentence_pairs)
        y_train = np.array(labels)

        text_path = DATA_PATH + 'QQP/dev.tsv'
        df = pd.read_csv(text_path, sep='\t', error_bad_lines=False)
        sentence_pairs = np.array(list(zip(df.question1.values, df.question2.values)))
        labels = df.is_duplicate.values

        X_test = np.array(sentence_pairs)
        y_test = np.array(labels)
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.current_seed)

        print(type(X_train), X_train.shape, X_test.shape)  #
        print(type(y_train), y_train.shape, y_test.shape)  #

        train_dst = (X_train, y_train)
        test_dst = (X_test, y_test)

    elif args.dataset == 'WNLI':
        text_path = DATA_PATH + 'WNLI/train.tsv'
        df = pd.read_csv(text_path, sep='\t')  # ,error_bad_lines=False)
        sentence_pairs = np.array(list(zip(df.sentence1.values, df.sentence2.values)))
        labels = df.label.values

        X_train = np.array(sentence_pairs)
        y_train = np.array(labels)

        text_path = DATA_PATH + 'WNLI/dev.tsv'
        df = pd.read_csv(text_path, sep='\t')  # ,error_bad_lines=False)
        sentence_pairs = np.array(list(zip(df.sentence1.values, df.sentence2.values)))
        labels = df.label.values

        X_test = np.array(sentence_pairs)
        y_test = np.array(labels)
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.current_seed)

        print(type(X_train), X_train.shape, X_test.shape)  #
        print(type(y_train), y_train.shape, y_test.shape)  #

        train_dst = (X_train, y_train)
        test_dst = (X_test, y_test)

    elif args.dataset == 'RTE':
        label_dict = {
            'entailment': 0,
            'not_entailment': 1
        }
        text_path = DATA_PATH + 'RTE/train.tsv'
        df = pd.read_csv(text_path, sep='\t', error_bad_lines=False)
        df = df.dropna()
        sentence_pairs = np.array(list(zip(df.sentence1.values, df.sentence2.values)))
        labels = df.label.values
        labels = [label_dict[_label] for _label in labels]

        X_train = np.array(sentence_pairs)
        y_train = np.array(labels)

        text_path = DATA_PATH + 'RTE/dev.tsv'
        df = pd.read_csv(text_path, sep='\t', error_bad_lines=False)
        df = df.dropna()
        sentence_pairs = np.array(list(zip(df.sentence1.values, df.sentence2.values)))
        labels = df.label.values
        labels = [label_dict[_label] for _label in labels]

        X_test = np.array(sentence_pairs)
        y_test = np.array(labels)
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.current_seed)

        print(type(X_train), X_train.shape, X_test.shape)  #
        print(type(y_train), y_train.shape, y_test.shape)  #

        train_dst = (X_train, y_train)
        test_dst = (X_test, y_test)

    elif args.dataset == 'MMLU':
        subject_list = []
        choices = ["A", "B", "C", "D"]
        args.label_dict = {0:'A', 1:'B', 2:'C', 3:'D'}


        train_set_file, test_set_file = get_dataset_path(args.model_list[str(index)])
        if train_set_file is None or test_set_file is None:
            train_set_file = DATA_PATH + 'MMLU/auxiliary_train/'
            test_set_file = DATA_PATH + 'MMLU/test/'
            dev_set_file = DATA_PATH + 'MMLU/dev/'



        ### train ###
        df_train = pd.DataFrame()
        for name in sorted(os.listdir(train_set_file))[:1]:
            # print(name[:-4])
            _df = pd.read_csv(train_set_file + name, header=None)[:10]
            #names=['prompt', 'A', 'B', 'C', 'D', 'answer'])  #
            answers = choices[:_df.shape[1]-2]
            # subject_name = name
            # _df['subject'] = subject_name
            # subject_list.append(subject_name)

            df_train = pd.concat([df_train, _df])

        prompt_list = []
        answer_list = []
        for i in range(len(df_train)):
            prompt_end = format_example(df_train, i, include_answer=False)
            # train_prompt = gen_prompt(dev_df, subject, k)
            prompt = prompt_end #train_prompt + prompt_end
            answer = df_train.iloc[i, df_train.shape[1]-1]
            # if 1:#df_train.iloc[i]['subject'] == args.subject:  # in subject_list:
            #     subject = df_train.iloc[i]['subject']
            #     prompt_head = "The following are multiple choice questions (with answers) about {}.\n\n".format(
            #         format_subject(subject))
            #     prompt_end, answer = format_example(df_train, i, include_answer=False)
            #     prompt = prompt_head + prompt_end
            prompt_list.append(prompt)
            answer_list.append(answer)
        X_train = np.array(prompt_list)
        y_train = np.array(answer_list)
        # print('---- Train -----')
        # print('Prompt:',prompt_list[0])
        # print('Label:',answer_list[0])


        ### test ###
        args.subject_list = sorted([f.split("_test.csv")[0] for f in os.listdir(test_set_file) if "_test.csv" in f])
        # print('all subjects:',args.subject_list)

        prompt_list = []
        answer_list = []
        ntrain = args.n_shot
        print('n_shot = ',ntrain)
        for subject in args.subject_list: # [args.subject]:
            dev_df = pd.read_csv(os.path.join(dev_set_file, subject + "_dev.csv"), header=None)[:ntrain]
            test_df = pd.read_csv(os.path.join(test_set_file, subject + "_test.csv"), header=None)
            answers = choices[:test_df.shape[1]-2]

            for i in range(test_df.shape[0]):
                # get prompt and make sure it fits
                k = ntrain
                prompt_end = format_example(test_df, i, include_answer=False)
                train_prompt = gen_prompt(dev_df, subject, k)
                prompt = train_prompt + prompt_end

                while crop(prompt) != prompt:
                    k -= 1
                    train_prompt = gen_prompt(dev_df, subject, k)
                    prompt = train_prompt + prompt_end

                label = test_df.iloc[i, test_df.shape[1]-1]

                prompt_list.append(prompt)
                answer_list.append(label)

        X_test = np.array(prompt_list)
        y_test = np.array(answer_list)
        # print('---- Test -----')
        # print('Prompt:',prompt_list[0])
        # print('Label:',answer_list[0])

        print('Data:')
        print(type(X_train), X_train.shape, X_test.shape)  #
        print(type(y_train), y_train.shape, y_test.shape)  #

        train_dst = (X_train, y_train)
        test_dst = (X_test, y_test)

    elif args.dataset == 'Lambada':
        # def create_chat_prompt(prompt, text):
        #     return [
        #         {"role": "system", "content": prompt},
        #         {"role": "user", "content": text}
        #     ]
        prompt = "Please complete the passages with the correct next word."
        # def create_chat_prompt(text, prompt=prompt):
        #     return [
        #         {"role": "system", "content": prompt},
        #         {"role": "user", "content": text}
        #     ]

        dataset_split = args.model_list[str(index)]
        if 'train_set_file' in dataset_split and 'test_set_file' in dataset_split:
            data_file = dataset_split['train_set_file']
        else:
            data_file = DATA_PATH + 'Lambada'
        print(data_file)

        dataset = load_dataset(data_file)

        doc_stride = args.doc_stride

        prompt_tokens = args.tokenizer.tokenize(prompt)  # .strip().split()
        max_seq_length = args.max_seq_length - len(prompt_tokens)

        ## train
        train_all_texts = dataset['train'][:]['text']
        train_domain = dataset['train'][:]['domain']
        texts = []
        target_word = []

        for _all_text in train_all_texts[:]:
            all_doc_tokens = args.tokenizer.tokenize(_all_text)#.strip().split()
            # all_doc_tokens = [c for c in all_doc_tokens if c not in string.punctuation]

            start_offset = 0
            while start_offset < len(all_doc_tokens):
                length = len(all_doc_tokens) - start_offset - 1  # max length left
                if length > max_seq_length:
                    length = max_seq_length

                text_tokens = all_doc_tokens[start_offset: start_offset + length]  # 0 1...7

                text = args.tokenizer.convert_tokens_to_string(prompt_tokens + text_tokens)
                last_word = all_doc_tokens[start_offset + length]
                # print('text:',text)
                # print('last_word:',last_word)

                # text = " ".join(text)

                # message = create_chat_prompt(text)
                # text = prompt+text #args.tokenizer.apply_chat_template(message, tokenize=False)

                texts.append(text)
                target_word.append(last_word)

                if start_offset + doc_stride + 1 >= len(all_doc_tokens) or \
                        start_offset + length + 1 >= len(all_doc_tokens):
                    break

                start_offset += min(length, doc_stride)

        X_train = np.array(texts)
        y_train = np.array(target_word)

        ## test
        test_all_texts = dataset['test'][:]['text']
        test_domain = dataset['test'][:]['domain']
        texts = []
        target_word = []
        for _all_text in test_all_texts[:]:
            all_doc_tokens = args.tokenizer.tokenize(_all_text)  # .strip().split()

            text_tokens = all_doc_tokens[:-1]

            # pad_length = args.max_seq_length - len(prompt_tokens) - len(text_tokens)

            # if pad_length > 0:
            #     text_tokens = [args.tokenizer.eos_token *pad_length] + prompt_tokens + text_tokens
            # else:
            text_tokens = prompt_tokens + text_tokens

            text = args.tokenizer.convert_tokens_to_string(text_tokens)

            texts.append(text)  # messages.append( )
            target_word.append(all_doc_tokens[-1])

        X_test = np.array(texts)
        y_test = target_word

        print('X_train:', len(X_train), '  X_test:', len(X_test))
        train_dst = (X_train, y_train)
        test_dst = (X_test, y_test)

    elif args.dataset == 'SQuAD':
        train_set_file, test_set_file = get_dataset_path(args.model_list[str(index)])
        if train_set_file is None or test_set_file is None:
            train_set_file = DATA_PATH + '/SQuAD/data/train-v1.1.json'
            test_set_file = DATA_PATH + '/SQuAD/data/dev-v1.1.json'

        max_seq_length = args.max_seq_length
        doc_stride = args.doc_stride
        max_query_length = args.max_query_length

        ## train
        train_examples = standard_read_squad_examples(input_file=train_set_file, is_training=True)
        train_features = convert_examples_to_features(train_examples, tokenizer=args.tokenizer, \
                                                      max_seq_length=max_seq_length, doc_stride=doc_stride, \
                                                      max_query_length=max_query_length, is_training=True)

        inputs = []
        labels = []
        for feature in train_features[:]:
            inputs.append(feature)
            labels.append([feature["start_position"], feature["end_position"]])

        X_train = inputs
        y_train = labels

        ## test
        test_examples = standard_read_squad_examples(input_file=test_set_file, is_training=False)
        test_features = convert_examples_to_features(test_examples, tokenizer=args.tokenizer,
                                                     max_seq_length=max_seq_length, doc_stride=doc_stride, \
                                                     max_query_length=max_query_length, is_training=False)
        # print('test_features:',len(test_features),test_features[0].keys())

        inputs = []
        labels = []
        for feature in test_features[:]:
            inputs.append(feature)
            labels.append([feature["start_position"], feature["end_position"]])

        X_test = inputs
        y_test = labels

        train_dst = (X_train, y_train)
        test_dst = (X_test, y_test)

        print('X:', type(X_train), len(X_train), len(X_test), type(X_train[0]))  #
        print('y', type(y_train), len(y_train), len(y_test), y_train[0])  #

    elif args.dataset == 'Alpaca':
        data_path = DATA_PATH + '/alpaca/alpaca_data.json'
        DEFAULT_PAD_TOKEN = "[PAD]"
        DEFAULT_EOS_TOKEN = "</s>"
        DEFAULT_BOS_TOKEN = "<s>"
        DEFAULT_UNK_TOKEN = "<unk>"
        IGNORE_INDEX = args.tokenizer.pad_token_id  # -100
        PROMPT_DICT = {
            "prompt_input": (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
            ),
            "prompt_no_input": (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Response:"
            ),
        }

        def _make_r_io_base(f, mode: str):
            if not isinstance(f, io.IOBase):
                f = open(f, mode=mode)
            return f

        def jload(f, mode="r"):
            """Load a .json file into a dictionary."""
            f = _make_r_io_base(f, mode)
            jdict = json.load(f)
            f.close()
            return jdict

        list_data_dict = jload(data_path)

        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]  # instructions
        # targets = [f"{example['output']}{args.tokenizer.eos_token}" for example in list_data_dict] # local
        targets = [f"{example['output']}" for example in list_data_dict]  # local

        X_data = sources#[:500] # list of instruction text
        y_data = targets#[:500] # list of answer text

        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.1,
                                                            random_state=args.current_seed)

        print('train data:', len(X_train), len(y_train))
        print('test data:', len(X_test), len(y_test))

        train_dst = (X_train, y_train)
        test_dst = (X_test, y_test)

        # input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        # input_ids = torch.nn.utils.rnn.pad_sequence(
        #     input_ids, batch_first=True, padding_value=args.tokenizer.pad_token_id
        # )
        # labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

    elif args.dataset == 'CodeAlpaca':
        data_path = DATA_PATH + '/CodeAlpaca-20k/code_alpaca_20k.json'
        DEFAULT_PAD_TOKEN = "[PAD]"
        DEFAULT_EOS_TOKEN = "</s>"
        DEFAULT_BOS_TOKEN = "<s>"
        DEFAULT_UNK_TOKEN = "<unk>"
        IGNORE_INDEX = args.tokenizer.pad_token_id  # -100
        PROMPT_DICT = {
            "prompt_input": (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
            ),
            "prompt_no_input": (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Response:"
            ),
        }

        def _make_r_io_base(f, mode: str):
            if not isinstance(f, io.IOBase):
                f = open(f, mode=mode)
            return f

        def jload(f, mode="r"):
            """Load a .json file into a dictionary."""
            f = _make_r_io_base(f, mode)
            jdict = json.load(f)
            f.close()
            return jdict

        list_data_dict = jload(data_path)

        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]  # instructions
        # targets = [f"{example['output']}{args.tokenizer.eos_token}" for example in list_data_dict] # local
        targets = [f"{example['output']}" for example in list_data_dict]  # local

        X_data = sources # list of instruction text
        y_data = targets # list of answer text

        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.1,
                                                            random_state=args.current_seed)

        print('train data:', len(X_train), len(y_train))
        print('test data:', len(X_test), len(y_test))

        train_dst = (X_train, y_train)
        test_dst = (X_test, y_test)

    elif args.dataset == 'GMS8K':
        data_path = DATA_PATH + '/GMS8K/'
        problem_prompt = ("Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: Let's think step by step.")
        def read_jsonl(path: str):
            with open(path) as fh:
                return [json.loads(line) for line in fh.readlines() if line]

        def get_final_ans(ans):
            temp_ans = ans.split('#### ')[1]
            temp_ans = int(temp_ans.replace(',', ''))
            return str(temp_ans)

        def get_examples(data_path, split):
            path = os.path.join(data_path, f"{split}.jsonl")
            examples = read_jsonl(path)

            for ex in examples:
                ex.update(question=ex["question"] + "\n")
                ex.update(answer=ex["answer"])

            print(f"{len(examples)} {split} examples")
            return examples

        ##### Train #####
        train_examples = get_examples(data_path, 'train') # list of [  {'quesion':... , 'answer':...} ...]
        X_train = np.array([ problem_prompt.format(instruction=_ex['question']+ "<|endoftext|>") for _ex in train_examples])
        y_train = np.array([ _ex['answer'] for _ex in train_examples])
        # print('TRAIN IN:')
        # print(X_train[0])
        # print('-'*100)
        # print('TRAIN ANS:')
        # print(y_train[0])

        ##### Test #####
        test_examples = get_examples(data_path, 'test') # list of [  {'quesion':... , 'answer':...} ...]
        X_test = np.array([ problem_prompt.format(instruction=_ex['question']) for _ex in test_examples])
        y_test = np.array([ get_final_ans(_ex['answer']) for _ex in test_examples])
        # print('TEST IN:')
        # print(X_test[0])
        # print('-'*100)
        # print('TEST ANS:')
        # print(y_test[0])
        # print('-'*100)

        train_dst = (X_train, y_train)
        test_dst = (X_test, y_test)

        print('X:',type(X_train), len(X_train), len(X_test))  #
        print('y',type(y_train), len(y_train), len(y_test))  #

    elif args.dataset == 'MATH':
        data_path = DATA_PATH + '/MATH/'
        problem_prompt = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
        )
        def remove_boxed(s):
            left = "\\boxed{"
            try:
                assert s[:len(left)] == left
                assert s[-1] == "}"
                return s[len(left):-1]
            except:
                return None

        def last_boxed_only_string(string):
            idx = string.rfind("\\boxed")
            if idx < 0:
                idx = string.rfind("\\fbox")
                if idx < 0:
                    return None

            i = idx
            right_brace_idx = None
            num_left_braces_open = 0
            while i < len(string):
                if string[i] == "{":
                    num_left_braces_open += 1
                if string[i] == "}":
                    num_left_braces_open -= 1
                    if num_left_braces_open == 0:
                        right_brace_idx = i
                        break
                i += 1

            if right_brace_idx == None:
                retval = None
            else:
                retval = string[idx:right_brace_idx + 1]

            return retval

        all_filenames = glob.glob(data_path+'/train/*/*.json')
        hendrycks_math_ins = []
        hendrycks_math_answers = []
        for fname in all_filenames[:]:
            with open(fname, 'r') as fp:
                try:
                    problem_data = json.load(fp)
                except Exception as e:
                    print(f"Error loading JSON from {fname}", e)
                    raise e
            temp_instr = problem_prompt.format(instruction=problem_data["problem"])
            hendrycks_math_ins.append("\nQUESTION:\n" + temp_instr + "\FULL SOLUTION:\n")

            temp_ans = problem_data['solution']
            # temp_ans = remove_boxed(last_boxed_only_string(temp_ans))
            hendrycks_math_answers.append(temp_ans)

        X_train = np.array(hendrycks_math_ins)
        y_train = np.array(hendrycks_math_answers)

        # print('TRAIN IN:')
        # print(hendrycks_math_ins[0])
        # print('-'*100)
        # print('TRAIN ANS:')
        # print(hendrycks_math_answers[0])


        all_filenames = glob.glob(data_path+'/test/*/*.json')
        hendrycks_math_ins = []
        hendrycks_math_answers = []
        for fname in all_filenames[:]:
            with open(fname, 'r') as fp:
                try:
                    problem_data = json.load(fp)
                except Exception as e:
                    print(f"Error loading JSON from {fname}", e)
                    raise e
            temp_instr = problem_prompt.format(instruction=problem_data["problem"])
            hendrycks_math_ins.append(temp_instr)

            temp_ans = problem_data['solution']
            # print('temp_ans:',temp_ans)
            # print('last_boxed_only_string(temp_ans):',last_boxed_only_string(temp_ans))
            temp_ans = remove_boxed(last_boxed_only_string(temp_ans))
            # print('final_ans:',temp_ans)
            # print('-'*100)

            hendrycks_math_answers.append(temp_ans)

        X_test = np.array(hendrycks_math_ins)
        y_test = np.array(hendrycks_math_answers)

        # print('TEST IN:')
        # print('sources:',hendrycks_math_ins[0])
        # print('-'*100)
        # print('TEST ANS:')
        # print('targets:',hendrycks_math_answers[0])
        # print('-'*100)

        train_dst = (X_train, y_train)
        test_dst = (X_test, y_test)

        print('X:',type(X_train), len(X_train), len(X_test))  #
        print('y',type(y_train), len(y_train), len(y_test))  #


    elif not args.dataset:
        return None
        
    elif args.dataset=='huanhuan':
        tokenizer=args.tokenizer
        dataset_split = args.model_list[str(index)]
        if 'train_set_file' in dataset_split and 'test_set_file' in dataset_split:
            data_path = dataset_split['train_set_file']

        def process_func(example):
            MAX_LENGTH = 128  # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
            input_ids, attention_mask, labels = [], [], []
            instruction = tokenizer(
                f"<|im_start|>system\n现在你要扮演皇帝身边的女人--甄嬛<|im_end|>\n<|im_start|>user\n{example['instruction'] + example['input']}<|im_end|>\n<|im_start|>assistant\n",
                add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
            response = tokenizer(f"{example['output']}", add_special_tokens=False)
            input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
            attention_mask = instruction["attention_mask"] + response["attention_mask"] + [
                1]  # 因为eos token咱们也是要关注的所以 补充为1
            labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
            if len(input_ids) > MAX_LENGTH:  # 做一个截断
                input_ids = input_ids[:MAX_LENGTH]
                attention_mask = attention_mask[:MAX_LENGTH]
                labels = labels[:MAX_LENGTH]
            else:
                while len(input_ids) < MAX_LENGTH:
                    input_ids.append(tokenizer.pad_token_id)
                    attention_mask.append(0)
                    labels.append(tokenizer.pad_token_id)
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

        def fine_tune_dataset():
            # 将JSON文件转换为CSV文件
            # df = pd.read_json(os.path.join(DATA_PATH, 'huanhuan.json'))
            df = pd.read_json(data_path)
            ds = Dataset.from_pandas(df)
            ds.shuffle(seed=SEED)
            ds = ds.map(process_func, remove_columns=ds.column_names)
            # ds.split
            ans = ds.train_test_split(test_size=0.1, seed=SEED)
            return ans['train'], ans['test']
        _dataset_train,_dataset_test=fine_tune_dataset()
        def _reformat_dataset(dataset):
            x=[]
            y=[]
            for _idx,i in enumerate(dataset):
                # if _idx>5:
                #     break
                x.append({})
                for k,v in i.items():
                    if k=='labels':
                        y.append(v)
                    else:
                        x[-1].update({k:v})

            return x,y
        train_dst=_reformat_dataset(_dataset_train)
        test_dst=_reformat_dataset(_dataset_test)

    else:
        assert args.dataset == 'news20', "dataset not supported yet"

    # train_dst = (train_dst[0].to(args.device),train_dst[1].to(args.device))
    # test_dst = (test_dst[0].to(args.device),test_dst[1].to(args.device))
    # if args.need_auxiliary == 1:
    #     aux_dst = (aux_dst[0].to(args.device),aux_dst[1].to(args.device))

    train_dst = dataset_partition_llm(args, index, train_dst, half_dim)
    test_dst = dataset_partition_llm(args, index, test_dst, half_dim)
    if args.need_auxiliary == 1:
        aux_dst = dataset_partition_llm(args, index, aux_dst, half_dim)

    # important
    if args.need_auxiliary == 1:
        return args, half_dim, train_dst, test_dst, aux_dst
    else:
        return args, half_dim, train_dst, test_dst


def get_dataset_path(dataset_split):
    if 'train_set_file' in dataset_split and 'test_set_file' in dataset_split:
        train_set_file = dataset_split['train_set_file']
        test_set_file = dataset_split['test_set_file']
        return train_set_file, test_set_file
    return None, None
