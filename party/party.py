import os
import sys
import numpy as np
import random

sys.path.append(os.pardir)

import torch
from torch.utils.data import DataLoader

from evaluates.attacks.attack_api import AttackerLoader
from evaluates.defenses.defense_api import DefenderLoader
from load.LoadDataset import load_dataset_per_party, load_dataset_per_party_backdoor, load_dataset_per_party_noisysample, load_unlearning_dataset_per_party
from load.LoadModels import load_models_per_party

from utils.noisy_label_functions import add_noise
from utils.noisy_sample_functions import noisy_sample
from utils.basic_functions import cross_entropy_for_onehot, tf_distance_cov_cor, pairwise_dist
from utils.communication_protocol_funcs import Cache

from sys import getsizeof


class Party(object):
    def __init__(self, args, index, need_data=True):
        self.name = "party#" + str(index + 1)
        self.index = index
        self.args = args
        # data for training and testing
        self.half_dim = -1
        self.train_data = None
        self.test_data = None
        self.aux_data = None
        self.train_label = None
        self.test_label = None
        self.aux_label = None
        self.train_attribute = None
        self.test_attribute = None
        self.aux_attribute = None
        self.train_dst = None
        self.test_dst = None
        self.aux_dst = None
        self.train_loader = None
        self.test_loader = None
        self.aux_loader = None
        self.attribute_loader = None
        self.attribute_iter = None
        self.local_batch_data = None
        # unlearning
        self.remove_indices = None
        # backdoor poison data and label and target images list
        self.train_poison_data = None
        self.train_poison_label = None
        self.test_poison_data = None
        self.test_poison_label = None
        self.train_target_list = None
        self.test_target_list = None
        # local model
        self.local_model = None
        self.local_model_optimizer = None
        # global_model
        self.global_model = None
        self.global_model_optimizer = None

        # attack and defense
        # self.attacker = None
        self.defender = None

        self.prepare_data(args, index)
        self.prepare_model(args, index)
        # self.prepare_attacker(args, index)
        # self.prepare_defender(args, index)

        self.local_gradient = None
        self.local_pred = None
        self.local_pred_clone = None
        self.ascent_gradient = None

        self.cache = Cache()
        self.prev_batches = []
        self.num_local_updates = 0
        # baseline
        self.regular_h = args.regular_h
        self.apply_R2S = args.apply_R2S
        self.velocity = [torch.zeros_like(param) for param in self.local_model.parameters()]
        self.momentum = 0.9  # 可以根据需要调整动量系数

    def receive_gradient(self, gradient):
        self.local_gradient = gradient
        return
    
    # def give_tmp_pred4unlearn_start(self):
    #     def restore_bias(layer, original_biases):
    #         for name, sub_layer in layer.named_children():
    #             if name in original_biases:
    #                 sub_layer.bias.data.copy_(original_biases[name])
    #             restore_bias(sub_layer, original_biases)
    #     def zero_bias(layer, original_biases):
    #         for name, sub_layer in layer.named_children():
    #             if hasattr(sub_layer, 'bias') and sub_layer.bias is not None:
    #                 original_biases[name] = sub_layer.bias.clone()
    #                 sub_layer.bias.data.zero_()
    #             zero_bias(sub_layer, original_biases)
    #     # 保存原始的 bias 值
    #     original_biases = {}
    #     zero_bias(self.local_model, original_biases) 
    #     # 计算输出
    #     self.local_model.eval()
    #     self.local_pred = self.local_model(self.local_batch_data)     
    #     # 打印模型参数
    #     # for name, param in self.local_model.named_parameters():
    #     #     print(name, param)
    #     # print(f"[debug] in party give_tmp_pred4unlearn_start, local_pred size:{self.local_pred.size()}, local_batch_data size:{self.local_batch_data.size()}")
    #     # print(f"[debug] in party give_tmp_pred4unlearn_start, local_pred:{self.local_pred}, local_batch_data:{self.local_batch_data}")
    #     restore_bias(self.local_model, original_biases)
    #     self.local_pred_clone = self.local_pred.detach().clone()
    #     self.local_model.train()
    #     return self.local_pred, self.local_pred_clone

    def give_pred(self):
        self.local_pred = self.local_model(self.local_batch_data)

        # ####### Noisy Sample #########
        # if self.args.apply_ns == True and (self.index in self.args.attack_configs['party']):
        #     assert 'noise_lambda' in self.args.attack_configs, 'need parameter: noise_lambda'
        #     assert 'noise_rate' in self.args.attack_configs, 'need parameter: noise_rate'
        #     assert 'party' in self.args.attack_configs, 'need parameter: party'
        #     noise_rate = self.args.attack_configs['noise_rate'] if ('noise_rate' in self.args.attack_configs) else 0.1
        #     noisy_list = []
        #     noisy_list = random.sample(range(self.local_pred.size()[0]), (int(self.local_pred.size()[0]*noise_rate)))
        #     scale = self.args.attack_configs['noise_lambda']

        #     self.local_batch_data[noisy_list] = noisy_sample(self.local_batch_data[noisy_list],scale)
        #     self.local_pred = self.local_model(self.local_batch_data)
        # ####### Noisy Sample #########

        # ####### Missing Feature #######
        if (self.args.apply_mf == True):
            assert 'missing_rate' in self.args.attack_configs, 'need parameter: missing_rate'
            assert 'party' in self.args.attack_configs, 'need parameter: party'
            missing_rate = self.args.attack_configs['missing_rate']

            if (self.index in self.args.attack_configs['party']):
                missing_list = random.sample(range(self.local_pred.size()[0]),
                                             (int(self.local_pred.size()[0] * missing_rate)))
                # print(f"[debug] in party: party{self.index}, missing list:", missing_list, len(missing_list))
                self.local_pred[missing_list] = torch.zeros(self.local_pred[missing_list].size()).to(self.args.device)
        # ####### Missing Feature #######

        self.local_pred_clone = self.local_pred.detach().clone()

        return self.local_pred, self.local_pred_clone

    def prepare_data(self, args, index, load_deleted=False):
        # prepare raw data for training
        if args.apply_backdoor == True:
            print("in party prepare_data, will prepare poison data for backdooring")
            (
                args,
                self.half_dim,
                (self.train_data, self.train_label),
                (self.test_data, self.test_label),
                (self.train_poison_data, self.train_poison_label),
                (self.test_poison_data, self.test_poison_label),
                self.train_target_list,
                self.test_target_list,
            ) = load_dataset_per_party_backdoor(args, index)
        if args.apply_ns == True:
            print("in party prepare_data, will prepare noisy data for NoisySampleBackdoor")
            (
                args,
                self.half_dim,
                (self.train_data, self.train_label),
                (self.test_data, self.test_label),
                (self.train_poison_data, self.train_poison_label),
                (self.test_poison_data, self.test_poison_label),
            ) = load_dataset_per_party_noisysample(args, index)
        elif args.need_auxiliary == 1:
            (
                args,
                self.half_dim,
                train_dst,
                test_dst,
                aux_dst
            ) = load_dataset_per_party(args, index)
            if len(train_dst) == 2:
                self.train_data, self.train_label = train_dst
                self.test_data, self.test_label = test_dst
                self.aux_data, self.aux_label = aux_dst
            elif len(train_dst) == 3:
                self.train_data, self.train_label, self.train_attribute = train_dst
                self.test_data, self.test_label, self.test_attribute = test_dst
                self.aux_data, self.aux_label, self.aux_attribute = aux_dst
            # print(f"in party load data, aux_data have length:{self.aux_data.shape}, train_data have length={self.train_data.shape}")
        elif (args.remove_specific_features or args.random_remove_features_percentage or args.remove_specific_clients or args.random_remove_samples_percentage or args.remove_specific_samples or args.random_remove_information_percentage or args.remove_specific_information):
            (
                args,
                self.half_dim,
                train_dst,
                test_dst,
                self.remove_indices
            ) = load_unlearning_dataset_per_party(args, index, load_deleted)
            if len(train_dst) == 2:
                self.train_data, self.train_label = train_dst
                self.test_data, self.test_label = test_dst
            elif len(train_dst) == 3:
                self.train_data, self.train_label, self.train_attribute = train_dst
                self.test_data, self.test_label, self.test_attribute = test_dst
        else:
            (
                args,
                self.half_dim,
                train_dst,
                test_dst,
            ) = load_dataset_per_party(args, index)
            if len(train_dst) == 2:
                self.train_data, self.train_label = train_dst
                self.test_data, self.test_label = test_dst
            elif len(train_dst) == 3:
                self.train_data, self.train_label, self.train_attribute = train_dst
                self.test_data, self.test_label, self.test_attribute = test_dst

    def prepare_data_loader(self, batch_size):
        self.train_loader = DataLoader(self.train_dst, batch_size=batch_size)  # , shuffle=True
        self.test_loader = DataLoader(self.test_dst, batch_size=batch_size)  # , shuffle=True
        if self.args.need_auxiliary == 1 and self.aux_dst != None:
            self.aux_loader = DataLoader(self.aux_dst, batch_size=batch_size)
        if self.train_attribute != None:
            self.attribute_loader = DataLoader(self.train_attribute, batch_size=batch_size)
            self.attribute_iter = iter(self.attribute_loader)

    def prepare_model(self, args, index):
        # prepare model and optimizer
        (
            args,
            self.local_model,
            self.local_model_optimizer,
            self.global_model,
            self.global_model_optimizer,
        ) = load_models_per_party(args, index)

    # def prepare_attacker(self, args, index):
    #     if index in args.attack_configs['party']:
    #         self.attacker = AttackerLoader(args, index, self.local_model)

    # def prepare_defender(self, args, index):
    #     if index in args.attack_configs['party']:
    #         self.defender = DefenderLoader(args, index)

    def give_current_lr(self):
        return (self.local_model_optimizer.state_dict()['param_groups'][0]['lr'])

    def LR_decay(self, i_epoch):
        eta_0 = self.args.main_lr
        eta_t = eta_0 / (np.sqrt(i_epoch + 1))
        for param_group in self.local_model_optimizer.param_groups:
            param_group['lr'] = eta_t

    def obtain_local_data(self, data):
        self.local_batch_data = data

    def local_forward():
        # args.local_model()
        pass

    # def local_backward(self):
    #     # update local model
    #     self.local_model_optimizer.zero_grad()
    #     # ########## for passive local mid loss (start) ##########
    #     # if passive party in defense party, do
    #     if (
    #         self.args.apply_mid == True
    #         and (self.index in self.args.defense_configs["party"])
    #         and (self.index < self.args.k - 1)
    #         ):
    #         # get grad for local_model.mid_model.parameters()
    #         self.local_model.mid_loss.backward(retain_graph=True)
    #         self.local_model.mid_loss = torch.empty((1, 1)).to(self.args.device)
    #     # ########## for passive local mid loss (end) ##########
    #     self.weights_grad_a = torch.autograd.grad(
    #         self.local_pred,
    #         self.local_model.parameters(),
    #         grad_outputs=self.local_gradient,
    #         retain_graph=True,
    #     )
    #     for w, g in zip(self.local_model.parameters(), self.weights_grad_a):
    #         if w.requires_grad:
    #             w.grad = g.detach()
    #     self.local_model_optimizer.step()

    def calculate_gradient4eva(self):
        self.weights_grad_a = torch.autograd.grad(
                    self.local_pred,
                    self.local_model.parameters(),
                    grad_outputs=self.local_gradient,
                    retain_graph=True
                )
        gradient, bias_gradient = self.weights_grad_a[0].flatten().detach().cpu().numpy(), self.weights_grad_a[1].flatten().detach().cpu().numpy()
        return gradient
    
    def calculate_gradient4ga(self):
        weights_grad_a = torch.autograd.grad(
                    self.local_pred,
                    self.local_model.parameters(),
                    grad_outputs=self.local_gradient,
                    retain_graph=True
                )
        if self.ascent_gradient is None:
            self.ascent_gradient = weights_grad_a
        else:
            self.ascent_gradient = [a + b for a, b in zip(self.ascent_gradient, weights_grad_a)]

    def local_backward(self, weight=None):
        self.num_local_updates += 1  # another update

        # update local model
        self.local_model_optimizer.zero_grad()
        # for w in self.local_model.parameters():
        #     if w.requires_grad:
        #         print("zero grad results in", w.grad) # None for all

        if self.regular_h:
            regularizer = self.local_pred_clone.sum(axis=0) * self.regular_h
            regularizer = regularizer.to(self.args.device)
        # print(f"[debug] in party local_backward, regularizer:{regularizer, regularizer.size()}")

        # ########## for passive local mid loss (start) ##########
        # if passive party in defense party, do
        if (
                self.args.apply_mid == True
                and (self.index in self.args.defense_configs["party"])
                and (self.index < self.args.k - 1)
        ):
            # get grad for local_model.mid_model.parameters()
            self.local_model.mid_loss.backward(retain_graph=True)
            self.local_model.mid_loss = torch.empty((1, 1)).to(self.args.device)
            # for w in self.local_model.parameters():
            #     if w.requires_grad:
            #         print("mid_loss grad results in", w.grad)
            # # get grad for local_model.local_model.parameters()
            # get grad for local_model.parameters()
            self.weights_grad_a = torch.autograd.grad(
                self.local_pred,
                # self.local_model.local_model.parameters(),
                self.local_model.parameters(),
                grad_outputs=self.local_gradient,
                retain_graph=True,
            )
            # for w, g in zip(self.local_model.local_model.parameters(), self.weights_grad_a):
            for w, g in zip(self.local_model.parameters(), self.weights_grad_a):
                if w.requires_grad:
                    if w.grad != None:
                        w.grad += g.detach()
                    else:
                        w.grad = g.detach()
            # for w in self.local_model.parameters():
            #     if w.requires_grad:
            #         print("total grad results in", w.grad)
        # ########## for passive local mid loss (end) ##########
        elif (
                self.args.apply_dcor == True
                and (self.index in self.args.defense_configs["party"])
                and (self.index < self.args.k - 1)  # pasive defense
        ):
            self.weights_grad_a = torch.autograd.grad(
                self.local_pred,
                self.local_model.parameters(),
                grad_outputs=self.local_gradient,
                retain_graph=True,
            )
            for w, g in zip(self.local_model.parameters(), self.weights_grad_a):
                # print('w:',w.size(),'g:',g.size())
                if w.requires_grad:
                    w.grad = g.detach()

            ########## dCor Loss ##########
            # print('dcor passive defense')
            self.distance_correlation_lambda = self.args.defense_configs['lambda']
            loss_dcor = self.distance_correlation_lambda * torch.log(
                tf_distance_cov_cor(self.local_pred, torch.flatten(self.local_batch_data, start_dim=1)))
            dcor_gradient = torch.autograd.grad(
                loss_dcor, self.local_model.parameters(), retain_graph=True, create_graph=True
            )
            # print('dcor_gradient:',len(dcor_gradient),dcor_gradient[0].size())
            for w, g in zip(self.local_model.parameters(), dcor_gradient):
                # print('w:',w.size(),'g:',g.size())
                if w.requires_grad:
                    w.grad += g.detach()
            ########## dCor Loss ##########
        elif (self.args.apply_adversarial == True and (self.index in self.args.defense_configs["party"])):
            # ########## adversarial training loss (start) ##########
            try:
                target_attribute = self.attribute_iter.__next__()
            except StopIteration:
                self.attribute_iter = iter(self.attribute_loader)
                target_attribute = self.attribute_iter.__next__()
            assert target_attribute.shape[0] == self.local_model.adversarial_output.shape[
                0], f"[Error] Data not aligned, target has shape: {target_attribute.shape}, pred has shape {self.local_model.adversarial_output.shape}"
            attribute_loss_fn = torch.nn.CrossEntropyLoss()
            attribute_loss = self.args.defense_configs["lambda"] * attribute_loss_fn(
                self.local_model.adversarial_output, target_attribute)
            attribute_loss.backward(retain_graph=True)
            self.local_model.adversarial_output = None
            self.weights_grad_a = torch.autograd.grad(
                self.local_pred,
                self.local_model.local_model.parameters(),
                # self.local_model.parameters(),
                grad_outputs=self.local_gradient,
                retain_graph=True,
            )
            for w, g in zip(self.local_model.local_model.parameters(), self.weights_grad_a):
                # for w, g in zip(self.local_model.parameters(), self.weights_grad_a):
                if w.requires_grad:
                    if w.grad != None:
                        w.grad += g.detach()
                    else:
                        w.grad = g.detach()
            # ########## adversarial training loss (end) ##########
        else:
            torch.autograd.set_detect_anomaly(True)
            if weight != None:  # CELU
                ins_batch_cached_grad = torch.mul(weight.unsqueeze(1), self.local_gradient)
                self.weights_grad_a = torch.autograd.grad(
                    self.local_pred,
                    self.local_model.parameters(),
                    grad_outputs=ins_batch_cached_grad,
                    retain_graph=True
                )
            else:
                self.weights_grad_a = torch.autograd.grad(
                    self.local_pred,
                    self.local_model.parameters(),
                    grad_outputs=self.local_gradient,
                    retain_graph=True
                )
            # SGDM
            if self.apply_R2S:
                # 将 self.weights_grad_a 转换为列表
                # print(f"[debug] in party local_backward, apply R2S, self.weights_grad_a:{self.weights_grad_a}")
                # print(f"[debug] in party local_backward, apply R2S, self.velocity:{self.velocity}")
                self.weights_grad_a = list(self.weights_grad_a)
                for i, grad in enumerate(self.weights_grad_a):
                    self.velocity[i] = self.momentum * self.velocity[i] + grad
                    self.weights_grad_a[i] = self.velocity[i]
                # 将 self.weights_grad_a 转换为元组
                self.weights_grad_a = tuple(self.weights_grad_a)
            if (self.ascent_gradient is not None) and self.args.apply_gradient_ascent:
                with torch.no_grad():
                    # print(f"[debug] in party local_backward, apply gradient ascent, self.ascent_gradient:{self.ascent_gradient}")
                    # print(f"[debug] in party local_backward, apply gradient ascent, self.weights_grad_a:{self.weights_grad_a}")
                    self.weights_grad_a = [a + b for a, b in zip(self.weights_grad_a, self.ascent_gradient)]
                    # print(f"[debug] in party local_backward, apply gradient ascent, self.weights_grad_a:{self.weights_grad_a}")
                    self.ascent_gradient = None

            for w, g in zip(self.local_model.parameters(), self.weights_grad_a):
                if w.requires_grad:
                    if self.regular_h:
                        # print(f"[debug] g size:{g.size()}, regularizer size:{regularizer.size()}")
                        # print(f"[debug] g:{g}, regularizer:{regularizer}")
                        if len(g.size()) != 1:
                            # 确保数据类型一致
                            w.grad = (g.detach() + regularizer.reshape(-1, 1)).to(g.detach())
                        else:
                            w.grad = g.detach().to(g.detach()) + regularizer.to(g.detach())
                    else:
                        w.grad = g.detach()
        self.local_model_optimizer.step()
        # print(f"[debug] self.weights_grad_a:{self.weights_grad_a}")
