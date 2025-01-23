import sys, os
sys.path.append(os.pardir)
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm import tqdm
import numpy as np
import random
import time
import copy

# from models.vision import resnet18, MLP2
from utils.basic_functions import cross_entropy_for_onehot, append_exp_res, multiclass_auc
from utils.communication_protocol_funcs import get_size_of

# from evaluates.attacks.attack_api import apply_attack
from evaluates.defenses.defense_api import apply_defense
from evaluates.defenses.defense_functions import *
from utils.constants import *
import utils.constants as shared_var
from utils.marvell_functions import KL_gradient_perturb
from utils.noisy_label_functions import add_noise
from utils.noisy_sample_functions import noisy_sample
from utils.communication_protocol_funcs import compress_pred,Cache,ins_weight
from evaluates.attacks.attack_api import AttackerLoader


tf.compat.v1.enable_eager_execution() 

    
def forget_the_unlearn_features(data, remove_indices):
    """
    data: torch.tensor
    remove_indices: list
    set the index in remove_indices to 0, keep others
    keep the order of the features
    """
    if len(remove_indices) == 0:
        return data
    else:
        data_np = data.detach().cpu().numpy()
        data_np[:, remove_indices] = 0
        return torch.tensor(data_np)
    

def forget_specific_information(data, remove_indices):
    """
    data: torch.tensor
    remove_indices: indices of the specific information
    set the index in remove_indices to 0, keep others
    keep the order of the features
    """
    if len(remove_indices) == 0:
        return data
    else:
        data_np = data.detach().cpu().numpy()
        data_np[remove_indices] = 0
        return torch.tensor(data_np)


class UnlearningTask(object):

    def __init__(self, args):
        self.args = args
        self.k = args.k
        self.device = args.device
        self.dataset_name = args.dataset
        # self.train_dataset = args.train_dst
        # self.val_dataset = args.test_dst
        # self.half_dim = args.half_dim
        self.epochs = args.main_epochs
        self.lr = args.main_lr
        self.batch_size = args.batch_size
        self.models_dict = args.model_list
        # self.num_classes = args.num_classes
        # self.num_class_list = args.num_class_list
        self.num_classes = args.num_classes
        self.exp_res_dir = args.exp_res_dir

        self.exp_res_path = args.exp_res_path
        self.parties = args.parties
        
        self.Q = args.Q # FedBCD

        self.parties_data = None
        self.gt_one_hot_label = None
        self.clean_one_hot_label  = None
        self.pred_list = []
        self.pred_list_clone = []
        self.pred_gradients_list = []
        self.pred_gradients_list_clone = []
        
        # FedBCD related
        self.local_pred_list = []
        self.local_pred_list_clone = []
        self.local_pred_gradients_list = []
        self.local_pred_gradients_list_clone = []
        
        self.loss = None
        self.train_acc = None
        self.flag = 1
        self.stopping_iter = 0
        self.stopping_time = 0.0
        self.stopping_commu_cost = 0
        self.communication_cost = 0
        self.total_time = 0.0

        # Early Stop
        self.early_stop_threshold = args.early_stop_threshold
        self.final_epoch = 0
        self.current_epoch = 0
        self.current_step = 0
        self.test_loss = 0.0
        self.train_loss_list = []
        self.train_acc_list = []

        # some state of VFL throughout training process
        self.first_epoch_state = None
        self.middle_epoch_state = None
        self.final_state = None
        # self.final_epoch_state = None # <-- this is save in the above parameters

        self.num_update_per_batch = args.num_update_per_batch
        self.num_batch_per_workset = args.Q #args.num_batch_per_workset
        self.max_staleness = self.num_update_per_batch*self.num_batch_per_workset 

        # for recording historical clients' pred
        self.save_distribute_percent = args.save_distribute_percent
        self.historical_clients_pred = []
        # self.tmp_clients_pred = []
        self.clients_pred = None

        # for unlearning dataset setting
        self.src_file_dir = args.src_file_dir
        self.unlearning_clients = args.unlearning_clients
        self.unlearning_features = args.unlearning_features
        self.unlearning_samples = args.unlearning_samples
        self.unlearning_classes = args.unlearning_classes
        self.unlearning_specific_information = args.unlearning_specific_information
        self.overall_pred = torch.tensor(args.overall_pred, dtype=torch.float64).to(self.device)
        self.delta_overall_pred = None
        self.distribute_percent = args.distributed_percent
        self.wth_DP = args.wth_DP
        self.dir_path = None
        try:
            self.clients_to_remove = args.clients_to_remove
        except:
            self.clients_to_remove = None
        if self.unlearning_features:
            self.features_to_remove = args.features_to_remove
        if self.unlearning_samples:
            self.samples_to_remove = args.samples_to_remove
        if self.unlearning_classes:
            self.classes_to_remove = args.classes_to_remove
        if self.unlearning_specific_information:
            self.information_to_remove = args.information_to_remove

        #for stream unlearning request
        self.unlearning_request_version = 0
        self.acumulate_overall_pred = None
        self.unlearning_version_list = [0 for _ in range(self.k)]
        self.online_clients = args.online_clients

        # for unlearning evaluation
        self.overall_gradient = []
        self.gradient_list = [[] for _ in range(self.k)]
        self.gradient_residue_sum = 0.0
        self.gradient_residue_sum_list = []
        self.apply_gradient_ascent = args.apply_gradient_ascent

        # for unlearning h history
        self.h_delta_list = []
        self.online_distribute_percent_list = []
        self.remaining_distribute_percent_list = []
        self.online_h_delta_list = []
        self.overall_pred_list = []
        self.unlearning_clients_h_delta_list = []
    
    def gradient_ascent(self):
        for ik in range(self.k):
            self.parties[ik].local_model.eval()
            self.parties[ik].local_model_optimizer.zero_grad()
        self.parties[self.k-1].global_model.eval()
        if self.parties[self.k-1].global_model_optimizer is not None:
            self.parties[self.k-1].global_model_optimizer.zero_grad()

        for ik in range(self.k):
            if self.unlearning_samples or self.unlearning_classes:
                self.parties[ik].prepare_data(self.args, ik, load_deleted=True)
            self.parties[ik].prepare_data_loader(batch_size=self.batch_size)
        data_loader_list = [self.parties[ik].train_loader for ik in range(self.k)]
        if self.unlearning_samples or self.unlearning_classes:  
            h_overall = self.overall_pred[self.samples_to_remove, :]
        else:
            h_overall = self.overall_pred
        start = 0
        end = self.batch_size if self.batch_size < h_overall.shape[0] else h_overall.shape[0]
        for parties_data in zip(*data_loader_list):
            # find the corresponding data in the delta_overall_pred
            h_overall_batch = h_overall[start:end]
            start = end
            end = start + self.batch_size if start + self.batch_size < h_overall.shape[0] else h_overall.shape[0]
            for ik in range(self.k):
                self.parties[ik].obtain_local_data(parties_data[ik][0])
            for ik in self.online_clients:
                pred, pred_detach = self.parties[ik].give_pred()
                if ik == (self.k-1): # Active party update local pred
                    pred_clone = torch.autograd.Variable(pred_detach, requires_grad=True).to(self.args.device)
                    self.parties[ik].update_local_pred(pred_clone)
                if ik < (self.k-1):
                    pred_clone = torch.autograd.Variable(pred_detach, requires_grad=True).to(self.args.device)
                    self.parties[self.k-1].receive_pred(pred_clone, ik) 
            gt_one_hot_label = self.label_to_one_hot(parties_data[self.k-1][1], self.num_classes).to(self.device)
            self.parties[self.k-1].gt_one_hot_label = gt_one_hot_label
            gradient_list = self.parties[self.k-1].unlearning_calculate_gradient(h_overall_batch, self.online_clients, test=False)
            if len(gradient_list)>1:
                for _i in range(len(gradient_list)-1):
                    self.communication_cost += get_size_of(gradient_list[_i+1])
            # active party update local gradient
            self.parties[self.k-1].update_local_gradient(gradient_list[-1])
            self.parties[self.k-1].calculate_gradient4ga()
            # active party transfer gradient to passive parties
            for ik in range(len(gradient_list)-1):
                self.parties[self.online_clients[ik]].receive_gradient(gradient_list[ik])
                self.parties[self.online_clients[ik]].calculate_gradient4ga()
        if self.unlearning_samples or self.unlearning_classes:
            for ik in range(self.k):
                self.parties[ik].prepare_data(self.args, ik, load_deleted=False)
                self.parties[ik].prepare_data_loader(batch_size=self.batch_size)

        # unlearned_samples_pred = self.overall_pred.detach().cpu().numpy()[self.samples_to_remove, :]
        # unlearned_samples_labels = torch.load('./configs/deleted_labels.pkl')
        # unlearned_samples_gt_one_hot_label = self.label_to_one_hot(unlearned_samples_labels, self.num_classes).to(self.device)
        # # only work when the global model is not trainable
        # pred_input = torch.autograd.Variable(torch.tensor(unlearned_samples_pred, dtype=torch.float32), requires_grad=True).to(self.device)
        # loss = self.parties[self.k-1].criterion(pred_input, unlearned_samples_gt_one_hot_label)
        # pred_gradients_list_clone = []
        # for ik in self.online_clients:
        #     gradient = torch.autograd.grad(loss, pred_input, retain_graph=True, create_graph=True)
        #     pred_gradients_list_clone.append(gradient[0].detach().clone().cpu().numpy())
        # self.ascent_gradient = np.array(pred_gradients_list_clone)
        # print("ascent_gradient shape:", self.ascent_gradient.shape) # (k, train_data_size, num_classes)
        for ik in range(self.k):
            self.parties[ik].local_model_optimizer.zero_grad()
        if self.parties[self.k-1].global_model_optimizer is not None:
            self.parties[self.k-1].global_model_optimizer.zero_grad()

    def calc_delta_overall_pred(self, unlearn_start=True):
        # =========calc_delta_overall_pred================
        for ik in range(self.k):
            self.parties[ik].prepare_data_loader(batch_size=self.batch_size)

        for ik in range(self.k):
            self.parties[ik].local_model.eval()
        self.parties[self.k-1].global_model.eval()

        with torch.no_grad():
            data_loader_list = [self.parties[ik].train_loader for ik in range(self.k)]
            pred_list = [[] for ik in range(self.k)]
            org_pred_list = [[] for ik in range(self.k)]
            start = 0
            end = self.batch_size if self.batch_size < self.overall_pred.shape[0] else self.overall_pred.shape[0]
            for parties_data in zip(*data_loader_list):
                # print(f"batch size: {parties_data[0][0].size()}")
                self.clients_load_unlearned_batch_data(parties_data, start, end)
                start = end
                end = end + self.batch_size if end + self.batch_size < self.overall_pred.shape[0] else self.overall_pred.shape[0]
                for ik in self.online_clients:
                    _pred, _pred_clone= self.parties[ik].give_pred()
                    pred_list[ik] += _pred_clone.detach().cpu().numpy().tolist()
            # print(f"pred_list shape: {np.array(pred_list).shape}") # (clients_num, train_data_size, num_classes)
                if unlearn_start:
                    for ik in range(self.k):
                        self.parties[ik].obtain_local_data(parties_data[ik][0])
                    for ik in self.clients_to_remove:
                        org_pred, org_pred_clone= self.parties[ik].give_pred()
                        org_pred_list[ik] += org_pred_clone.detach().cpu().numpy().tolist()
            if unlearn_start:
                for ik in self.clients_to_remove:
                    if self.delta_overall_pred == None:
                        self.delta_overall_pred = np.array(pred_list[ik]) - np.array(org_pred_list[ik])
                        # print(f"delta_overall_pred shape: {self.delta_overall_pred.shape}", self.delta_overall_pred) # (train_data_size, num_classes)
                    else:
                        self.delta_overall_pred += np.array(pred_list[ik]) - np.array(org_pred_list[ik])
                    self.communication_cost += get_size_of(torch.tensor(pred_list[ik])) #MB
                self.overall_pred = self.overall_pred + torch.tensor(self.delta_overall_pred, dtype=torch.float64).to(self.device)
                # torch.save(self.delta_overall_pred, self.dir_path + 'delta_overall_pred.pkl')
                # raise NotImplementedError("unlearn_start is True, not implemented yet")
            else:
                return [np.array(pred_list[ik]) for ik in range(self.k)]

            # check the delta_overall_pred if there is any nan or inf
            # if np.isnan(self.delta_overall_pred).any() or np.isinf(self.delta_overall_pred).any():
            #     print("[error] delta_overall_pred has nan or inf")
            #     index = np.isnan(self.delta_overall_pred) | np.isinf(self.delta_overall_pred)
            #     print(f"index: {index}")
            #     print(f"delta_overall_pred: {self.delta_overall_pred[index]}")
            #     exit(0)
            # if torch.isnan(self.overall_pred).any() or torch.isinf(self.overall_pred).any():
            #     print("[error] delta_overall_pred has nan or inf")
            #     index = torch.isnan(self.overall_pred) | torch.isinf(self.overall_pred)
            #     print(f"index: {index}")
            #     print(f"overall_pred: {self.overall_pred[index]}")
            #     exit(0)
            # print(f"delta_overall_pred shape: {self.delta_overall_pred.shape}", self.delta_overall_pred) # (train_data_size, num_classes)
            # print(f"overall_pred shape: {self.overall_pred.shape}", self.overall_pred) # (train_data_size, num_classes)
        # =========calc_delta_overall_pred==================

        # if self.acumulate_overall_pred == None:
        #     self.acumulate_overall_pred = self.delta_overall_pred
        # else:
        #     self.acumulate_overall_pred += self.delta_overall_pred
        # self.unlearning_request_version += 1
        # for ik in self.clients_to_remove:
        #     self.unlearning_version_list[ik] = self.unlearning_request_version

    def calculate_gradient_residue(self):
        self.overall_gradient = []
        for ik in range(self.k):
            grad = self.gradient_list[ik]
            if len(grad) == 0:
                continue
            self.overall_gradient.append(grad)
        self.overall_gradient = np.concatenate(self.overall_gradient)

    def unlearning_retrain_based_on_h(self):
        if self.args.apply_gradient_ascent:
            self.gradient_ascent()
        if self.unlearning_clients or self.unlearning_features or self.unlearning_specific_information:
            self.calc_delta_overall_pred(unlearn_start=True)
        elif self.unlearning_samples or self.unlearning_classes:
            # delete the self.samples_to_remove samples from the overall_pred
            self.overall_pred = np.delete(self.overall_pred.detach().cpu().numpy(), self.samples_to_remove, axis=0)
            self.overall_pred = torch.tensor(self.overall_pred, dtype=torch.float64).to(self.device)
        print_every = 1
        # Early Stop
        last_residue = 1000000
        last_train_loss = 1000000
        early_stop_count = 0

        data_loader_list = [self.parties[ik].train_loader for ik in range(self.k)]
        self.total_time = 0.0
        self.clients_pred = self.calc_delta_overall_pred(unlearn_start=False)

        for ik in range(self.k):
            self.parties[ik].local_model.train()
        self.parties[self.k-1].global_model.train()

        for i_epoch in range(self.epochs):
            # self.tmp_clients_pred = []
            start = 0
            end = self.batch_size if self.batch_size < self.overall_pred.shape[0] else self.overall_pred.shape[0]
            self.loss = 0.0
            suc_cnt = 0
            self.gradient_list = [[] for _ in range(self.k)]

            for parties_data in zip(*data_loader_list):
                # find the corresponding data in the delta_overall_pred
                h_overall_batch = self.overall_pred[start:end]
                enter_time = time.time()
                batch_loss, batch_suc_cnt = self.unlearning_retrain_batch_based_on_h(parties_data, h_overall_batch, start, end)
                exit_time = time.time()
                start = end
                end = end + self.batch_size if end + self.batch_size < self.overall_pred.shape[0] else self.overall_pred.shape[0]
                self.total_time += (exit_time-enter_time)
                self.loss += batch_loss
                suc_cnt += batch_suc_cnt
                # self.num_total_comms = self.num_total_comms + 1
            self.train_acc = suc_cnt / self.overall_pred.shape[0]
            self.train_loss_list.append(self.loss)
            self.train_acc_list.append(self.train_acc)
            self.calculate_gradient_residue()
            self.gradient_residue_sum = np.sqrt(np.dot(self.overall_gradient.T, self.overall_gradient))
            self.gradient_residue_sum_list.append(self.gradient_residue_sum)
            if i_epoch % print_every == 0:
                print(f"epoch {i_epoch}, train_loss: {self.loss}, train_acc: {self.train_acc}, gradient_residue_sum: {self.gradient_residue_sum}")
                # print("gradient:", self.overall_gradient)

            online_distribute_percent = np.array([])
            online_h_delta = np.array([])
            remaining_distribute_percent = np.array([])
            unlearn_client_delta = np.array([])
            cliens_pred_after = self.calc_delta_overall_pred(unlearn_start=False)

            for ik in range(self.k):
                if ik not in self.clients_to_remove:
                    if len(remaining_distribute_percent) == 0:
                        remaining_distribute_percent = self.distribute_percent[ik].copy()
                    else:
                        remaining_distribute_percent += self.distribute_percent[ik]
                    if ik in self.online_clients:
                        if len(online_distribute_percent) == 0:
                            online_distribute_percent = self.distribute_percent[ik].copy()
                        else:
                            online_distribute_percent += self.distribute_percent[ik]
                        if len(online_h_delta) == 0:
                            online_h_delta = np.array(cliens_pred_after[ik] - self.clients_pred[ik])
                        else:
                            online_h_delta += np.array(cliens_pred_after[ik] - self.clients_pred[ik])
                elif not self.unlearning_clients:
                    if len(unlearn_client_delta) == 0:
                        unlearn_client_delta = np.array(cliens_pred_after[ik] - self.clients_pred[ik])
                    else:
                        unlearn_client_delta += np.array(cliens_pred_after[ik] - self.clients_pred[ik])
                else:
                    unlearn_client_delta = np.zeros(self.clients_pred[ik].shape)
            
            # print('shape of remaining_distribute_percent:', np.array(remaining_distribute_percent).shape, remaining_distribute_percent)
            # print('shape of online_distribute_percent:', np.array(online_distribute_percent).shape, online_distribute_percent)
            # print('shape of online_h_delta:', np.array(online_h_delta).shape, online_h_delta)
            self.clients_pred = cliens_pred_after
            h_delta = online_h_delta / online_distribute_percent * remaining_distribute_percent
            index_divide_zero = np.where(online_distribute_percent == 0)
            h_delta[index_divide_zero] = 0
            if not self.unlearning_clients:
                h_delta += unlearn_client_delta
            # print('number of divide zero:', len(index_divide_zero[0]))
            # h_delta = np.array(cliens_pred_after[k-1] - self.clients_pred[k-1]) / self.distribute_percent[self.k-1] * remaining_distribute_percent
            self.overall_pred = self.overall_pred + torch.tensor(h_delta, dtype=torch.float64).to(self.device)
            # check the overall_pred, h_delta, online_distribute_percent, remaining_distribute_percent, online_h_delta if there is any nan or inf
            if np.isnan(h_delta).any() or np.isinf(h_delta).any():
                print("[error] h_delta has nan or inf")
                index = np.isnan(h_delta) | np.isinf(h_delta)
                # print which ordinal number has nan or inf
                print(f"index: {np.where(index)}")
                print(f"h_delta: {h_delta[index]}, online_distribute_percent: {online_distribute_percent[index]}, remaining_distribute_percent: {remaining_distribute_percent[index]}, online_h_delta: {online_h_delta[index]}")
                print(f"online_distribute_percent: {self.distribute_percent[0][index], self.distribute_percent[1][index], self.distribute_percent[2][index], self.distribute_percent[3][index]}")
                exit(0)
            # print('shape of h_delta:', h_delta.shape, h_delta)
            # print('shape of overall_pred:', self.overall_pred.shape, self.overall_pred)
            self.h_delta_list.append(h_delta)
            self.online_distribute_percent_list.append(online_distribute_percent)
            self.remaining_distribute_percent_list.append(remaining_distribute_percent)
            self.online_h_delta_list.append(online_h_delta)
            self.overall_pred_list.append(self.overall_pred)
            self.unlearning_clients_h_delta_list.append(unlearn_client_delta)

            # collect historical clients' pred
            if self.save_distribute_percent:
                self.historical_clients_pred.append(self.clients_pred)

            # LR decay
            # if i_epoch < 10:
            self.LR_Decay(i_epoch)

            #early stop by gradient residue
            
            if (self.gradient_residue_sum >= last_residue * 1.05 and not self.args.apply_R2S and not self.wth_DP) or self.loss >= last_train_loss:
                early_stop_count += 1
            else:
                self.trained_models = self.save_state(i_epoch)
                self.final_epoch = i_epoch
                early_stop_count = 0
            
            if early_stop_count >= self.early_stop_threshold:
                print('Early Stop at Epoch {}'.format(i_epoch))
                print('Final Epoch is {}'.format(self.final_epoch))
                break

            last_residue = min(self.gradient_residue_sum, last_residue)
            last_train_loss = min(self.loss, last_train_loss)

        self.final_state = self.save_state(self.final_epoch)
        self.final_state.update(self.save_party_data()) 
        # save trained models
        self.save_trained_models() 
        if self.args.save_h_history:
            self.save_h()
        print('Total time for unlearning retrain:', self.total_time)
        print('Total communication cost for unlearning retrain:', self.communication_cost)
        print("min_gradient_residue_sum:", last_residue)   
        return self.total_time, self.communication_cost, self.final_epoch

    def unlearning_retrain_batch_based_on_h(self, parties_data, h_overall, start, end):

        # ===========give gt_one_hot_label to active party================
        self.gt_one_hot_label = self.label_to_one_hot(parties_data[self.k-1][1], self.num_classes)
        self.gt_one_hot_label = self.gt_one_hot_label.to(self.device)
        encoder = self.args.encoder
        if self.args.apply_cae:
            assert encoder != None, "[error] encoder is None for CAE"
            _, gt_one_hot_label = encoder(self.gt_one_hot_label) 
            # _, test_one_hot_label = encoder(torch.tensor([[0.0,1.0],[1.0,0.0]]).to(self.args.device))
            # print("one hot label for DCAE 1.0 of 2 class", test_one_hot_label)   
            # for DCAE-1.0-2class, <[0.0,1.0],[1.0,0.0]> ==> <[0.9403, 0.0597],[0.0568, 0.9432]>        
        else:
            gt_one_hot_label = self.gt_one_hot_label
        self.parties[self.k-1].gt_one_hot_label = gt_one_hot_label
        # ===========give gt_one_hot_label to active party================

        self.clients_load_unlearned_batch_data(parties_data, start, end)
        for ik in self.online_clients:
            pred, pred_detach = self.parties[ik].give_pred()
            if ik == (self.k-1): # Active party update local pred
                pred_clone = torch.autograd.Variable(pred_detach, requires_grad=True).to(self.args.device)
                self.parties[ik].update_local_pred(pred_clone)
            if ik < (self.k-1):
                pred_clone = torch.autograd.Variable(pred_detach, requires_grad=True).to(self.args.device)
                self.parties[self.k-1].receive_pred(pred_clone, ik) 
            
        gradient_list = self.parties[self.k-1].unlearning_calculate_gradient(h_overall, self.online_clients, test=False)

        if self.wth_DP:
            # gradient_list = KL_gradient_perturb(gradient_list)
            gradient_list = AddNoise(self.args, gradient_list, sigma=1e-6)#=0.0001
            # print('gradient_list:', gradient_list)
        if len(gradient_list)>1:
            for _i in range(len(gradient_list)-1):
                self.communication_cost += get_size_of(gradient_list[_i+1])#MB
        # active party update local gradient
        self.parties[self.k-1].update_local_gradient(gradient_list[-1])
        self.parties[self.k-1].local_backward()
        self.parties[self.k-1].global_backward()
        # if self.save_distribute_percent:
        #     self.tmp_clients_pred.append(self.parties[self.k-1].save_received_pred())
        gradient_tmp, bias_gradient_tmp = self.parties[self.k-1].weights_grad_a[0].flatten().detach().cpu().numpy(), self.parties[self.k-1].weights_grad_a[1].flatten().detach().cpu().numpy()
        # gradient_tmp = np.concatenate((gradient_tmp, bias_gradient_tmp))
        if len(self.gradient_list[self.k-1]) == 0:
            self.gradient_list[self.k-1] = gradient_tmp
        else:
            self.gradient_list[self.k-1] += gradient_tmp
        
        # active party transfer gradient to passive parties
        for ik in range(len(gradient_list)-1):
            self.parties[self.online_clients[ik]].receive_gradient(gradient_list[ik])
            self.parties[self.online_clients[ik]].local_backward()
            gradient_tmp1, bias_gradient_tmp1 = self.parties[self.online_clients[ik]].weights_grad_a[0].flatten().detach().cpu().numpy(), self.parties[self.online_clients[ik]].weights_grad_a[1].flatten().detach().cpu().numpy()
            # if ik in self.clients_to_remove:
            #     print('gradient_tmp1:', gradient_tmp1)
            #     print('bias_gradient_tmp1:', bias_gradient_tmp1)
            # gradient_tmp1 = np.concatenate((gradient_tmp1, bias_gradient_tmp1))
            if len(self.gradient_list[self.online_clients[ik]]) == 0:
                self.gradient_list[self.online_clients[ik]] = gradient_tmp1
            else:
                self.gradient_list[self.online_clients[ik]] += gradient_tmp1
                    
        # ###### Noisy Label Attack #######
        # convert back to clean label to get true acc
        if self.args.apply_nl==True:
            real_batch_label = self.clean_one_hot_label
        else:
            real_batch_label = self.gt_one_hot_label
        # ###### Noisy Label Attack #######
        
        pred = self.parties[self.k-1].global_pred
        loss = self.parties[self.k-1].global_loss
        predict_prob = F.softmax(pred, dim=-1)
        # if self.args.apply_cae:
        #     predict_prob = encoder.decode(predict_prob)

        suc_cnt = torch.sum(torch.argmax(predict_prob, dim=-1) == torch.argmax(real_batch_label, dim=-1)).item()
        return loss.item(), suc_cnt
    
    def clients_load_unlearned_batch_data(self, parties_data, start=0, end=0):
        for ik in range(self.k):
            local_data = parties_data[ik][0]
            if (ik not in self.clients_to_remove) or self.unlearning_samples or self.unlearning_classes:
                local_data = local_data.to(self.device)
            if self.unlearning_features and (ik in self.clients_to_remove):
                local_data = forget_the_unlearn_features(local_data, self.features_to_remove).to(self.device)
            if self.unlearning_specific_information and (ik in self.clients_to_remove):
                # the first dimension of the information_to_remove is the row index, second dimension is the column index
                mask = (self.information_to_remove[0] >= start) & (self.information_to_remove[0] < end)
                filtered_indices = (self.information_to_remove[0][mask] - start, self.information_to_remove[1][mask])
                # print(f"filtered_indices: {filtered_indices}")
                local_data = forget_specific_information(local_data, filtered_indices).to(self.device)
            if self.unlearning_clients and (ik in self.clients_to_remove):
                local_data = torch.zeros(local_data.size()).to(self.device)
            self.parties[ik].obtain_local_data(local_data)
    
    def baseline_VFULR(self):
        self.calc_delta_overall_pred(unlearn_start=True)
        data_loader_list = [self.parties[ik].train_loader for ik in range(self.k)]
        for ik in range(self.k):
            self.parties[ik].local_model.train()
        self.parties[self.k-1].global_model.train()

        start = 0
        end = self.batch_size if self.batch_size < self.overall_pred.shape[0] else self.overall_pred.shape[0]
        self.loss = 0.0
        suc_cnt = 0
        self.gradient_list = [[] for _ in range(self.k)]
        start_time = time.time()
        for parties_data in zip(*data_loader_list):
            # find the corresponding data in the delta_overall_pred
            h_overall_batch = self.overall_pred[start:end]
            # ===========give gt_one_hot_label to active party================
            self.gt_one_hot_label = self.label_to_one_hot(parties_data[self.k-1][1], self.num_classes)
            self.gt_one_hot_label = self.gt_one_hot_label.to(self.device)
            encoder = self.args.encoder
            if self.args.apply_cae:
                assert encoder != None, "[error] encoder is None for CAE"
                _, gt_one_hot_label = encoder(self.gt_one_hot_label) 
                # _, test_one_hot_label = encoder(torch.tensor([[0.0,1.0],[1.0,0.0]]).to(self.args.device))
                # print("one hot label for DCAE 1.0 of 2 class", test_one_hot_label)   
                # for DCAE-1.0-2class, <[0.0,1.0],[1.0,0.0]> ==> <[0.9403, 0.0597],[0.0568, 0.9432]>        
            else:
                gt_one_hot_label = self.gt_one_hot_label
            self.parties[self.k-1].gt_one_hot_label = gt_one_hot_label
            # ===========give gt_one_hot_label to active party================

            self.clients_load_unlearned_batch_data(parties_data, start, end)
            start = end
            end = end + self.batch_size if end + self.batch_size < self.overall_pred.shape[0] else self.overall_pred.shape[0]
            for ik in self.online_clients:
                pred, pred_detach = self.parties[ik].give_pred()
                if ik == (self.k-1): # Active party update local pred
                    pred_clone = torch.autograd.Variable(pred_detach, requires_grad=True).to(self.args.device)
                    self.parties[ik].update_local_pred(pred_clone)
                if ik < (self.k-1):
                    pred_clone = torch.autograd.Variable(pred_detach, requires_grad=True).to(self.args.device)
                    self.parties[self.k-1].receive_pred(pred_clone, ik) 
            
            gradient_list = self.parties[self.k-1].unlearning_calculate_gradient(h_overall_batch, self.online_clients, test=False)
            for ik in range(self.k):
                self.parties[ik].local_pred_clone = - torch.tensor(self.delta_overall_pred, dtype=torch.float64).to(self.device)

            if self.wth_DP:
                # gradient_list = KL_gradient_perturb(gradient_list)
                gradient_list = AddNoise(self.args, gradient_list, sigma=0.0001)
                # print('gradient_list:', gradient_list)
            if len(gradient_list)>1:
                for _i in range(len(gradient_list)-1):
                    self.communication_cost += get_size_of(gradient_list[_i+1])#MB
            # active party update local gradient
            self.parties[self.k-1].update_local_gradient(gradient_list[-1])
            self.parties[self.k-1].local_backward()
            self.parties[self.k-1].global_backward()
            gradient_tmp, bias_gradient_tmp = self.parties[self.k-1].weights_grad_a[0].flatten().detach().cpu().numpy(), self.parties[self.k-1].weights_grad_a[1].flatten().detach().cpu().numpy()
            # gradient_tmp = np.concatenate((gradient_tmp, bias_gradient_tmp))
            if len(self.gradient_list[self.k-1]) == 0:
                self.gradient_list[self.k-1] = gradient_tmp
            else:
                self.gradient_list[self.k-1] += gradient_tmp
            
            # active party transfer gradient to passive parties
            for ik in range(len(gradient_list)-1):
                self.parties[self.online_clients[ik]].receive_gradient(gradient_list[ik])
                self.parties[self.online_clients[ik]].local_backward()
                gradient_tmp1, bias_gradient_tmp1 = self.parties[self.online_clients[ik]].weights_grad_a[0].flatten().detach().cpu().numpy(), self.parties[self.online_clients[ik]].weights_grad_a[1].flatten().detach().cpu().numpy()
                # if ik in self.clients_to_remove:
                #     print('gradient_tmp1:', gradient_tmp1)
                #     print('bias_gradient_tmp1:', bias_gradient_tmp1)
                # gradient_tmp1 = np.concatenate((gradient_tmp1, bias_gradient_tmp1))
                if len(self.gradient_list[self.online_clients[ik]]) == 0:
                    self.gradient_list[self.online_clients[ik]] = gradient_tmp1
                else:
                    self.gradient_list[self.online_clients[ik]] += gradient_tmp1
                        
            # ###### Noisy Label Attack #######
            # convert back to clean label to get true acc
            if self.args.apply_nl==True:
                real_batch_label = self.clean_one_hot_label
            else:
                real_batch_label = self.gt_one_hot_label
            # ###### Noisy Label Attack #######
            
            pred = self.parties[self.k-1].global_pred
            loss = self.parties[self.k-1].global_loss
            predict_prob = F.softmax(pred, dim=-1)
            # if self.args.apply_cae:
            #     predict_prob = encoder.decode(predict_prob)

            suc_cnt += torch.sum(torch.argmax(predict_prob, dim=-1) == torch.argmax(real_batch_label, dim=-1)).item()
            self.loss += loss.item()
        end_time = time.time()
        self.total_time = end_time - start_time
        self.train_acc = suc_cnt / self.overall_pred.shape[0]
        self.calculate_gradient_residue()
        self.gradient_residue_sum = np.sqrt(np.dot(self.overall_gradient.T, self.overall_gradient))
        # save trained models
        self.gradient_residue_sum_list.append(self.gradient_residue_sum)
        self.trained_models = self.save_state(epoch=0)
        self.save_trained_models()   
        # print('Total time for unlearning retrain:', self.total_time)
        # print('Total communication cost for unlearning retrain:', self.communication_cost)
        print("gradient_residue_sum:", self.gradient_residue_sum) 
        return self.total_time, self.communication_cost, 1

    def label_to_one_hot(self, target, num_classes=10):
        # print('label_to_one_hot:', target, type(target))
        try:
            _ = target.size()[1]
            # print("use target itself", target.size())
            onehot_target = target.type(torch.float32).to(self.device)
        except:
            target = torch.unsqueeze(target, 1).to(self.device)
            # print("use unsqueezed target", target.size())
            onehot_target = torch.zeros(target.size(0), num_classes, device=self.device)
            onehot_target.scatter_(1, target, 1)
        return onehot_target

    def LR_Decay(self,i_epoch):
        for ik in range(self.k):
            self.parties[ik].LR_decay(i_epoch)
        self.parties[self.k-1].global_LR_decay(i_epoch)
        
    def save_state(self, epoch):

        return {
            "online_clients": self.online_clients,
            "unlearning_clients": self.clients_to_remove,
            "finetune_epoch": self.final_epoch,
            "Total epoch": epoch,
            "Total communication cost": self.communication_cost,
            "Total time": self.total_time,
            "gradient_residue": self.gradient_residue_sum_list[-1],
            "model": [copy.deepcopy(self.parties[ik].local_model) for ik in self.online_clients],
            "global_model":copy.deepcopy(self.parties[self.args.k-1].global_model),
            # type(model) = <class 'xxxx.ModelName'>
            "model_names": [str(type(self.parties[ik].local_model)).split('.')[-1].split('\'')[-2] for ik in range(self.args.k)]+[str(type(self.parties[self.args.k-1].global_model)).split('.')[-1].split('\'')[-2]]
        }
    
    def save_party_data(self):
        return {
            "train_data": [copy.deepcopy(self.parties[ik].train_data) for ik in range(self.k)],
            "test_data": [copy.deepcopy(self.parties[ik].test_data) for ik in range(self.k)],
            "train_label": [copy.deepcopy(self.parties[ik].train_label) for ik in range(self.k)],
            "test_label": [copy.deepcopy(self.parties[ik].test_label) for ik in range(self.k)]
        }
    
    def determine_save_dir(self):
        if 'DP_test' in self.src_file_dir:
            dir_path = self.exp_res_dir + self.src_file_dir
        elif self.args.stream_unlearning:
            dir_path = self.exp_res_dir + self.src_file_dir.replace('retrain', 'Stream_Unlearning')
        elif self.args.regular_h:
            dir_path = self.exp_res_dir + self.src_file_dir.replace('retrain', 'VFULR')
        elif self.args.apply_R2S:
            dir_path = self.exp_res_dir + self.src_file_dir.replace('retrain', 'VFUFR')
        elif self.args.wth_asynchronous_unlearning and (not self.wth_DP) and (not self.apply_gradient_ascent):
            dir_path = self.exp_res_dir + self.src_file_dir.replace('retrain', 'Asynchronous_Unlearning')
        elif (not self.args.wth_asynchronous_unlearning) and (not self.wth_DP) and (not self.apply_gradient_ascent):
            dir_path = self.exp_res_dir + self.src_file_dir.replace('retrain', 'Batch_Unlearning')
        elif self.wth_DP and  self.args.wth_asynchronous_unlearning and (not self.apply_gradient_ascent):
            dir_path = self.exp_res_dir + self.src_file_dir.replace('retrain', 'Asynchronous_Unlearning_DP')
        elif self.wth_DP and (not self.args.wth_asynchronous_unlearning) and (not self.apply_gradient_ascent):
            dir_path = self.exp_res_dir + self.src_file_dir.replace('retrain', 'Batch_Unlearning_DP')
        elif self.args.wth_asynchronous_unlearning and (not self.wth_DP) and self.apply_gradient_ascent:
            dir_path = self.exp_res_dir + self.src_file_dir.replace('retrain', 'Asynchronous_Unlearning_GA')
        elif (not self.args.wth_asynchronous_unlearning) and (not self.wth_DP) and self.apply_gradient_ascent:
            dir_path = self.exp_res_dir + self.src_file_dir.replace('retrain', 'Batch_Unlearning_GA')
        elif self.wth_DP and  self.args.wth_asynchronous_unlearning and self.apply_gradient_ascent:
            dir_path = self.exp_res_dir + self.src_file_dir.replace('retrain', 'Asynchronous_Unlearning_DP_GA')
        elif self.wth_DP and (not self.args.wth_asynchronous_unlearning) and self.apply_gradient_ascent:
            dir_path = self.exp_res_dir + self.src_file_dir.replace('retrain', 'Batch_Unlearning_DP_GA')
        else:
            print('not implemented yet')
            print('please check the determine_save_dir function')
        self.dir_path = dir_path
        print('save trained models to:', dir_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    def save_trained_models(self):
        # dir_path = self.exp_res_dir + f'trained_models/parties{self.k}_topmodel{self.args.apply_trainable_layer}_epoch{self.epochs}/'
        dir_path = self.dir_path
        if self.args.apply_defense:
            file_path = dir_path + f'Unlearning_model_{self.args.defense_name}_{self.args.defense_configs}.pkl'
        else:
            file_path = dir_path + 'Unlearning_model.pkl'
        torch.save((self.trained_models), file_path)
        if self.save_distribute_percent:
            torch.save(self.historical_clients_pred, dir_path + 'historical_clients_pred.pkl')

        # draw the loss curve and accuracy curve
        plt.plot(self.train_loss_list, label='train_loss')
        plt.legend()
        plt.savefig(os.path.join(dir_path, 'train_loss.png'))
        plt.close()
        plt.plot(self.train_acc_list, label='train_acc')
        plt.legend()
        plt.savefig(os.path.join(dir_path, 'train_acc.png'))
        plt.close()
        plt.plot(self.gradient_residue_sum_list, label='gradient_residue_sum')
        plt.legend()
        plt.savefig(os.path.join(dir_path, 'gradient_residue_sum.png'))
        plt.close()
        torch.save([self.train_loss_list, self.train_acc_list, self.gradient_residue_sum_list], dir_path + 'train_process.pkl')

    def save_h(self):
        dir_path = self.dir_path
        file_path = dir_path + 'h_history.pkl'
        saved_h = {
            'h_delta_list': self.h_delta_list,
            'online_distribute_percent_list': self.online_distribute_percent_list,
            'remaining_distribute_percent_list': self.remaining_distribute_percent_list,
            'online_h_delta_list': self.online_h_delta_list,
            'overall_pred_list': self.overall_pred_list,
            'unlearning_clients_h_delta_list': self.unlearning_clients_h_delta_list
        }
        torch.save(saved_h, file_path)

    def evaluate_attack(self):
        self.attacker = AttackerLoader(self, self.args)
        if self.attacker != None:
            attack_acc = self.attacker.attack()
        return attack_acc

    def launch_defense(self, gradients_list, _type):
        
        if _type == 'gradients':
            return apply_defense(self.args, _type, gradients_list)
        elif _type == 'pred':
            return apply_defense(self.args, _type, gradients_list)
        else:
            # further extention
            return gradients_list

    def calc_label_recovery_rate(self, dummy_label, gt_label):
        success = torch.sum(torch.argmax(dummy_label, dim=-1) == torch.argmax(gt_label, dim=-1)).item()
        total = dummy_label.shape[0]
        return success / total
    
    def load_trained_model(self, model_path):
        trained_model = torch.load(model_path)
        online_clients = trained_model["online_clients"]
        unlearning_clients = trained_model["unlearning_clients"]
        finetune_epoch = trained_model["finetune_epoch"]
        Total_epoch = trained_model["Total epoch"]
        Total_communication_cost = trained_model["Total communication cost"]
        Total_time = trained_model["Total time"]
        gradient_residue = trained_model["gradient_residue"]
        model = trained_model["model"]
        global_model = trained_model["global_model"]
        model_names = trained_model["model_names"]
        # print("online_clients:", online_clients)
        # print("unlearning_clients:", unlearning_clients)
        # print("finetune_epoch:", finetune_epoch)
        # print("Total_epoch_for_unlearning_retrain:", Total_epoch_for_unlearning_retrain)
        # print("Total_communication_cost_for_unlearning_retrain:", Total_communication_cost_for_unlearning_retrain)
        # print("gradient_residue:", gradient_residue)
        # print("model:", model)
        # print("global_model:", global_model)
        # print("model_names:", model_names)
        self.final_epoch = finetune_epoch
        self.communication_cost = Total_communication_cost
        self.total_time = Total_time
        for ik in online_clients:
            self.parties[ik].local_model.load_state_dict(model[online_clients.index(ik)].state_dict())
            self.parties[ik].prepare_data_loader(batch_size=self.batch_size)
        self.parties[self.k-1].global_model.load_state_dict(global_model.state_dict())
        self.final_state = trained_model
        self.trained_models = trained_model
        self.final_state.update(self.save_party_data()) 

        # trained_model = torch.load(model_path)
        # online_clients = trained_model["online_clients"]
        # model = trained_model["model"]
        # global_model = trained_model["global_model"]
        # for ik in online_clients:
        #     self.parties[ik].local_model.load_state_dict(model[online_clients.index(ik)].state_dict())
        #     self.parties[ik].prepare_data_loader(batch_size=self.batch_size)
        # self.parties[self.k-1].global_model.load_state_dict(global_model.state_dict())
    
    def validate_test_dataset(self):
        # for ik in range(self.k):
        #     self.parties[ik].prepare_data_loader(batch_size=self.batch_size)

        for ik in range(self.k):
            self.parties[ik].local_model.eval()
        self.parties[self.k-1].global_model.eval()
        suc_cnt = 0
        sample_cnt = 0
        noise_suc_cnt = 0
        noise_sample_cnt = 0
        test_preds = []
        test_targets = []
        self.test_loss = 0.0

        # file_path_list = ['exp_result/adult_income\Q1/0/stored5/unlearn_client/1/trained_models\parties4_topmodel0_epoch400/Unlearning_model.pkl', 'exp_result/adult_income\Q1/0/stored5/unlearn_client/2/trained_models\parties4_topmodel0_epoch400/Unlearning_model.pkl']

        # clients_model = [0 for _ in range(self.k)]

        # if method == 'asynchronous':
        #     for _ in range(len(file_path_list)):
        #         model, model_names, online_clients, unlearning_clients, finetune_epoch, Total_epoch_for_unlearning_retrain, Total_communication_cost_for_unlearning_retrain, gradient_residue, global_model = self.load_unlearned_model(file_path_list[_])
        #         for __ in online_clients:
        #             if clients_model[__] == 0:
        #                 self.parties[__].local_model.load_state_dict(model[online_clients.index(__)].state_dict())
        #                 clients_model[__] = 1
        #         self.parties[self.k-1].global_model.load_state_dict(global_model.state_dict())
        # elif method == 'retrain':
        #     file_path = './configs/Retrain.pkl'
        #     state_dicts, model_names, distributed_percent, overall_pred = torch.load(file_path)
        #     for ik in range(self.k):
        #         self.parties[ik].local_model.load_state_dict(state_dicts[ik])
        # for i in range(len(self.online_clients)): 
        #     self.parties[self.online_clients[i]].local_model.load_state_dict(self.trained_models["model"][i].state_dict())
        # self.parties[self.k-1].global_model.load_state_dict(self.trained_models["global_model"].state_dict())

        with torch.no_grad():
            data_loader_list = [self.parties[ik].test_loader for ik in range(self.k)]
            start = 0
            end = self.batch_size if self.batch_size < self.overall_pred.shape[0] else self.overall_pred.shape[0]
            for parties_data in zip(*data_loader_list):
                gt_val_one_hot_label = self.label_to_one_hot(parties_data[self.k-1][1], self.num_classes)
                gt_val_one_hot_label = gt_val_one_hot_label.to(self.device)

                pred_list = []
                noise_pred_list = [] # for ntb attack
                missing_list_total = []

                for ik in range(self.k):
                    local_data = parties_data[ik][0]
                    if (ik not in self.clients_to_remove) or self.unlearning_samples or self.unlearning_classes or self.unlearning_specific_information:
                        local_data = local_data.to(self.device)
                    if self.unlearning_clients and (ik in self.clients_to_remove):
                        local_data = torch.zeros(local_data.size()).to(self.device)
                    if self.unlearning_features and (ik in self.clients_to_remove):
                        local_data = forget_the_unlearn_features(local_data, self.features_to_remove).to(self.device)
                    # if self.unlearning_specific_information:
                    #     # the first dimension of the information_to_remove is the row index, second dimension is the column index
                    #     mask = (self.information_to_remove[0] >= start) & (self.information_to_remove[0] < end)
                    #     filtered_indices = (self.information_to_remove[0][mask] - start, self.information_to_remove[1][mask])
                    #     local_data = forget_specific_information(local_data, filtered_indices).to(self.device)
                    _local_pred = self.parties[ik].local_model(local_data)
                    start = end
                    end = end + self.batch_size if end + self.batch_size < self.overall_pred.shape[0] else self.overall_pred.shape[0]
                    
                    ####### missing feature attack ######
                    if (self.args.apply_mf == True):
                        assert 'missing_rate' in self.args.attack_configs, 'need parameter: missing_rate'
                        assert 'party' in self.args.attack_configs, 'need parameter: party'
                        missing_rate = self.args.attack_configs['missing_rate']
                        
                        if (ik in self.args.attack_configs['party']):
                            missing_list = random.sample(range(_local_pred.size()[0]), (int(_local_pred.size()[0]*missing_rate)))
                            missing_list_total = missing_list_total + missing_list
                            _local_pred[missing_list] = torch.zeros(_local_pred[0].size()).to(self.args.device)
                            # print("[debug] in main VFL:", _local_pred[missing_list])
                        
                        pred_list.append(_local_pred)
                        noise_pred_list.append(_local_pred[missing_list])
                    ####### missing feature attack ######

                    else:
                        pred_list.append(_local_pred)

                # Normal Evaluation
                test_logit, test_loss = self.parties[self.k-1].aggregate(pred_list, gt_val_one_hot_label, test="True")
                self.test_loss += test_loss
                enc_predict_prob = F.softmax(test_logit, dim=-1)
                if self.args.apply_cae == True:
        
                    dec_predict_prob = self.args.encoder.decode(enc_predict_prob)
                        
                    test_preds.append(list(dec_predict_prob.detach().cpu().numpy()))
                    predict_label = torch.argmax(dec_predict_prob, dim=-1)
                else:
                    test_preds.append(list(enc_predict_prob.detach().cpu().numpy()))
                    predict_label = torch.argmax(enc_predict_prob, dim=-1)

                actual_label = torch.argmax(gt_val_one_hot_label, dim=-1)
                sample_cnt += predict_label.shape[0]
                suc_cnt += torch.sum(predict_label == actual_label).item()
                test_targets.append(list(gt_val_one_hot_label.detach().cpu().numpy()))

                # Evaluation on noised data in NTB
                if self.args.apply_mf == True: 
                    missing_list = list(set(missing_list_total))
                    noise_sample_cnt += len(missing_list)
                    noise_suc_cnt += torch.sum(predict_label[missing_list] == actual_label[missing_list]).item()
                    # print(f"this epoch, noise sample count is {len(missing_list)}, correct noise sample count is {torch.sum(predict_label[missing_list] == actual_label[missing_list]).item()}")
                    # noise_gt_val_one_hot_label = gt_val_one_hot_label[missing_list]

                    # noise_test_logit, noise_test_loss = self.parties[self.k-1].aggregate(noise_pred_list, noise_gt_val_one_hot_label, test="True")
                    # noise_enc_predict_prob = F.softmax(noise_test_logit, dim=-1)
                    # if self.args.apply_cae == True:
                    #     noise_dec_predict_prob = self.args.encoder.decode(noise_enc_predict_prob)
                    #     noise_predict_label = torch.argmax(noise_dec_predict_prob, dim=-1)
                    # else:
                    #     noise_predict_label = torch.argmax(noise_enc_predict_prob, dim=-1)

                    # noise_actual_label = torch.argmax(noise_gt_val_one_hot_label, dim=-1)
                    # noise_sample_cnt += noise_predict_label.shape[0]
                    # noise_suc_cnt += torch.sum(noise_predict_label == noise_actual_label).item()
                # elif self.args.apply_ns == True:
                #     noise_gt_val_one_hot_label = gt_val_one_hot_label[noisy_list]

                #     noise_test_logit, noise_test_loss = self.parties[self.k-1].aggregate(noise_pred_list, noise_gt_val_one_hot_label, test="True")
                #     noise_enc_predict_prob = F.softmax(noise_test_logit, dim=-1)
                #     if self.args.apply_cae == True:
                #         noise_dec_predict_prob = self.args.encoder.decode(noise_enc_predict_prob)
                #         noise_predict_label = torch.argmax(noise_dec_predict_prob, dim=-1)
                #     else:
                #         noise_predict_label = torch.argmax(noise_enc_predict_prob, dim=-1)

                #     noise_actual_label = torch.argmax(noise_gt_val_one_hot_label, dim=-1)
                #     noise_sample_cnt += noise_predict_label.shape[0]
                #     noise_suc_cnt += torch.sum(noise_predict_label == noise_actual_label).item()

            self.noise_test_acc = noise_suc_cnt / float(noise_sample_cnt) if noise_sample_cnt>0 else None
            self.test_acc = suc_cnt / float(sample_cnt)
            test_preds = np.vstack(test_preds)
            test_targets = np.vstack(test_targets)
            self.test_auc = np.mean(multiclass_auc(test_targets, test_preds))

            # print('test_loss:{:.4f}'.format(self.test_loss))
            # print('test_acc:{:.2f}'.format(self.test_acc))
            # print('test_auc:{:.2f}'.format(self.test_auc))
            if self.noise_test_acc != None:
                print('noisy_sample_acc:{:.2f}'.format(self.noise_test_acc))
            
        return test_preds, self.test_loss, self.test_acc, self.test_auc
    
    def fidelity_evaluation(self):
        # print('asynchronous')
        # test_preds, test_loss, test_acc, test_auc = self.validate_test_dataset('asynchronous')
        # print('retrain')
        # retrain_test_preds, retrain_test_loss, retrain_test_acc, retrain_test_auc = self.validate_test_dataset('retrain')
        # distance = F.kl_div(torch.tensor(test_preds).to(self.device).reshape(-1, self.num_classes).log(), torch.tensor(retrain_test_preds).to(self.device).reshape(-1, self.num_classes), reduction='batchmean').item()

        # print('distance:{}'.format(distance))
        # return distance, test_loss, test_acc, test_auc, retrain_test_loss, retrain_test_acc, retrain_test_auc
        if 'DP_test' in self.src_file_dir:
            retrain_dir_path = self.exp_res_dir + 'trained_models/DP'
        else:
            retrain_dir_path = self.exp_res_dir + self.src_file_dir
        retrain_test_preds = np.load(retrain_dir_path + 'test_preds.npy')
        dir_path = self.dir_path
        epsilon = 1e-10
        test_preds, test_loss, test_acc, test_auc = self.validate_test_dataset()
        np.save(dir_path + 'test_preds.npy', test_preds)
        test_preds = torch.tensor(test_preds, dtype=torch.float64).to(self.device).reshape(-1, self.num_classes)
        retrain_test_preds = torch.tensor(retrain_test_preds, dtype=torch.float64).to(self.device).reshape(-1, self.num_classes)
        test_preds = torch.clamp(test_preds, min=1e-10)
        distance = F.kl_div(test_preds.log()+epsilon, retrain_test_preds+epsilon, reduction='batchmean').item()
        # print('distance:{}'.format(distance))
        return distance, test_loss, test_acc, test_auc
    
    def efficacy_evaluation(self):
        # for ik in range(self.k):
        #     self.parties[ik].prepare_data_loader(batch_size=self.batch_size)
        # file_path_list = ['exp_result/adult_income\Q1/0/stored5/unlearn_client/1/trained_models\parties4_topmodel0_epoch400/Unlearning_model.pkl', 'exp_result/adult_income\Q1/0/stored5/unlearn_client/2/trained_models\parties4_topmodel0_epoch400/Unlearning_model.pkl']
        # clients_model = [0 for _ in range(self.k)]
        # for _ in range(len(file_path_list)):
        #     model, model_names, online_clients, unlearning_clients, finetune_epoch, Total_epoch_for_unlearning_retrain, Total_communication_cost_for_unlearning_retrain, gradient_residue, global_model = self.load_unlearned_model(file_path_list[_])
        #     for __ in online_clients:
        #         if clients_model[__] == 0:
        #             self.parties[__].local_model.load_state_dict(model[online_clients.index(__)].state_dict())
        #             clients_model[__] = 1
        #     self.parties[self.k-1].global_model.load_state_dict(global_model.state_dict())

        # file_path = './configs/Retrain.pkl' #NoDefense.pkl
        # state_dicts, model_names, distributed_percent, overall_pred = torch.load(file_path)
        # for ik in range(self.k):
        #     self.parties[ik].local_model.load_state_dict(state_dicts[ik])
        
        # for ik in range(self.k):
        #     self.parties[ik].local_model.train()
        # self.parties[self.k-1].global_model.train()
        # for i in range(len(self.online_clients)): 
        #     self.parties[self.online_clients[i]].local_model.load_state_dict(self.trained_models["model"][i].state_dict())
        # self.parties[self.k-1].global_model.load_state_dict(self.trained_models["global_model"].state_dict())
        
        data_loader_list = [self.parties[ik].train_loader for ik in range(self.k)]
        self.gradient_list = [[] for _ in range(self.k)]
        start = 0
        end = self.batch_size if self.batch_size < self.overall_pred.shape[0] else self.overall_pred.shape[0]

        for parties_data in zip(*data_loader_list):
            self.gt_one_hot_label = self.label_to_one_hot(parties_data[self.k-1][1], self.num_classes)
            self.gt_one_hot_label = self.gt_one_hot_label.to(self.device)
            self.parties_data = parties_data
            self.train_batch_only_for_efficacy_eva(self.parties_data,self.gt_one_hot_label, start, end)
            start = end
            end = end + self.batch_size if end + self.batch_size < self.overall_pred.shape[0] else self.overall_pred.shape[0]
       
        self.calculate_gradient_residue()
        self.gradient_residue_sum = np.sqrt(np.dot(self.overall_gradient.T, self.overall_gradient))
        # efficacy_eva_pred = self.calc_delta_overall_pred(unlearn_start=False)
        # torch.save(efficacy_eva_pred, self.dir_path + 'efficacy_eva_pred.pkl')
        # print("gradient_residue_sum:", self.gradient_residue_sum)
        # print("gradient:", self.overall_gradient)
        return self.gradient_residue_sum, self.overall_gradient
        
    def train_batch_only_for_efficacy_eva(self, parties_data, batch_label, start=0, end=0):
        '''
        batch_label: self.gt_one_hot_label   may be noisy
        '''
        encoder = self.args.encoder
        if self.args.apply_cae:
            assert encoder != None, "[error] encoder is None for CAE"
            _, gt_one_hot_label = encoder(batch_label) 
            # _, test_one_hot_label = encoder(torch.tensor([[0.0,1.0],[1.0,0.0]]).to(self.args.device))
            # print("one hot label for DCAE 1.0 of 2 class", test_one_hot_label)   
            # for DCAE-1.0-2class, <[0.0,1.0],[1.0,0.0]> ==> <[0.9403, 0.0597],[0.0568, 0.9432]>        
        else:
            gt_one_hot_label = batch_label
        
        self.parties[self.k-1].gt_one_hot_label = gt_one_hot_label
        # allocate data to each party
        self.clients_load_unlearned_batch_data(parties_data, start, end)

        # ======== Commu ===========
        if self.args.communication_protocol in ['Vanilla','FedBCD_p','Quantization','Topk'] or self.Q ==1 : # parallel FedBCD & noBCD situation
            for q in range(self.Q):
                if q == 0: 
                    # exchange info between parties
                    self.pred_transmit() 
                    self.gradient_transmit() 
                    # update parameters for all parties
                    for ik in range(self.k):
                        # if ik not in self.online_clients:
                        #     continue
                        #calculate SGD gradient##########
                        self.parties[ik].apply_R2S = False
                        self.parties[ik].regular_h = False
                        #################################
                        if len(self.gradient_list[ik]) == 0:
                            self.gradient_list[ik] = self.parties[ik].calculate_gradient4eva()
                        else:
                            self.gradient_list[ik] += self.parties[ik].calculate_gradient4eva()

                else: # FedBCD: additional iterations without info exchange
                    # for passive party, do local update without info exchange
                    for ik in range(self.k-1):
                        _pred, _pred_clone= self.parties[ik].give_pred() 
                        self.parties[ik].calculate_gradient_residue4eva() 
                    # for active party, do local update without info exchange
                    _pred, _pred_clone = self.parties[self.k-1].give_pred() 
                    _gradient = self.parties[self.k-1].give_gradient()
                    self.parties[self.k-1].calculate_gradient_residue4eva()

        elif self.args.communication_protocol in ['CELU']:
            if self.save_distribute_percent:
                print("[error] save_distribute_percent is not supported in CELU")
            for q in range(self.Q):
                if (q == 0) or (batch_label.shape[0] != self.args.batch_size): 
                    # exchange info between parties
                    self.pred_transmit() 
                    self.gradient_transmit() 
                    # update parameters for all parties
                    for ik in range(self.k):
                        self.parties[ik].local_backward()
                    self.parties[self.k-1].global_backward()

                    if (batch_label.shape[0] == self.args.batch_size): # available batch to cache
                        for ik in range(self.k):
                            batch = self.num_total_comms # current batch id
                            self.parties[ik].cache.put(batch, self.parties[ik].local_pred,\
                                self.parties[ik].local_gradient, self.num_total_comms + self.parties[ik].num_local_updates)
                else: 
                    for ik in range(self.k):
                        # Sample from cache
                        batch, val = self.parties[ik].cache.sample(self.parties[ik].prev_batches)
                        batch_cached_pred, batch_cached_grad, \
                            batch_cached_at, batch_num_update \
                                = val
                        
                        _pred, _pred_detach = self.parties[ik].give_pred()
                        weight = ins_weight(_pred_detach,batch_cached_pred,self.args.smi_thresh) # ins weight
                        
                        # Using this batch for backward
                        if (ik == self.k-1): # active
                            self.parties[ik].update_local_gradient(batch_cached_grad)
                            self.parties[ik].local_backward(weight)
                            self.parties[ik].global_backward()
                        else:
                            self.parties[ik].receive_gradient(batch_cached_grad)
                            self.parties[ik].local_backward(weight)


                        # Mark used once for this batch + check staleness
                        self.parties[ik].cache.inc(batch)
                        if (self.num_total_comms + self.parties[ik].num_local_updates - batch_cached_at >= self.max_staleness) or\
                            (batch_num_update + 1 >= self.num_update_per_batch):
                            self.parties[ik].cache.remove(batch)
                        
            
                        self.parties[ik].prev_batches.append(batch)
                        self.parties[ik].prev_batches = self.parties[ik].prev_batches[1:]#[-(num_batch_per_workset - 1):]
                        self.parties[ik].num_local_updates += 1

        elif self.args.communication_protocol in ['FedBCD_s']: # Sequential FedBCD_s
            if self.save_distribute_percent:
                print("[error] save_distribute_percent is not supported in FedBCD_s")

            for q in range(self.Q):
                if q == 0: 
                    #first iteration, active party gets pred from passsive party
                    self.pred_transmit() 
                    _gradient = self.parties[self.k-1].give_gradient()
                    if len(_gradient)>1:
                        for _i in range(len(_gradient)-1):
                            self.communication_cost += get_size_of(_gradient[_i+1])#MB
                    # active party: update parameters 
                    self.parties[self.k-1].local_backward()
                    self.parties[self.k-1].global_backward()
                else: 
                    # active party do additional iterations without info exchange
                    self.parties[self.k-1].give_pred()
                    _gradient = self.parties[self.k-1].give_gradient()
                    self.parties[self.k-1].local_backward()
                    self.parties[self.k-1].global_backward()

            # active party transmit grad to passive parties
            self.gradient_transmit() 

            # passive party do Q iterations
            for _q in range(self.Q):
                for ik in range(self.k-1): 
                    _pred, _pred_clone= self.parties[ik].give_pred() 
                    self.parties[ik].local_backward() 
        else:
            assert 1>2 , 'Communication Protocol not provided'
        # ============= Commu ===================
    
    def pred_transmit(self): # Active party gets pred from passive parties
        for ik in range(self.k):
            pred, pred_detach = self.parties[ik].give_pred()

            # defense applied on pred
            if self.args.apply_defense == True and self.args.apply_dp == True :
                # Only add noise to pred when launching FR attack(attaker_id=self.k-1)
                if (ik in self.args.defense_configs['party']) and (ik != self.k-1): # attaker won't defend its own attack
                    pred_detach = torch.tensor(self.launch_defense(pred_detach, "pred")) 
                # else:
                #     print(self.args.attack_type)

            if ik == (self.k-1): # Active party update local pred
                pred_clone = torch.autograd.Variable(pred_detach, requires_grad=True).to(self.args.device)
                self.parties[ik].update_local_pred(pred_clone)
            
            if ik < (self.k-1): # Passive party sends pred for aggregation
                ########### communication_protocols ###########
                if self.args.communication_protocol in ['Quantization','Topk']:
                    pred_detach = compress_pred( self.args ,pred_detach , self.parties[ik].local_gradient,\
                                    self.current_epoch, self.current_step).to(self.args.device)
                ########### communication_protocols ###########
                pred_clone = torch.autograd.Variable(pred_detach, requires_grad=True).to(self.args.device)
                
                self.communication_cost += get_size_of(pred_clone) #MB
                
                self.parties[self.k-1].receive_pred(pred_clone, ik) 
    
    def gradient_transmit(self):  # Active party sends gradient to passive parties
        gradient = self.parties[self.k-1].give_gradient() # gradient_clone

        if len(gradient)>1:
            for _i in range(len(gradient)-1):
                self.communication_cost += get_size_of(gradient[_i+1])#MB

        # defense applied on gradients
        if self.args.apply_defense == True and self.args.apply_dcor == False and self.args.apply_mid == False and self.args.apply_cae == False:
            if (self.k-1) in self.args.defense_configs['party']:
                gradient = self.launch_defense(gradient, "gradients")   
        if self.args.apply_dcae == True:
            if (self.k-1) in self.args.defense_configs['party']:
                gradient = self.launch_defense(gradient, "gradients")  
            
        # active party update local gradient
        self.parties[self.k-1].update_local_gradient(gradient[self.k-1])
        
        # active party transfer gradient to passive parties
        for ik in range(self.k-1):
            self.parties[ik].receive_gradient(gradient[ik])
        return
    
    def save4stream_unlearning(self):
        torch.save(self.overall_pred, self.dir_path + 'overall_pred.pkl')
        try:
            unlearning_set = torch.load(self.dir_path + 'unlearning_set.pkl')
        except:
            unlearning_set = {
                "online_clients": self.online_clients,
                "unlearning_clients": [],
                "unlearning_samples": [],
                "unlearning_features": [],
                "unlearning_information": [],
                "unlearning_classes": []}
        
        if self.unlearning_clients:
            unlearning_set["unlearning_clients"] = self.clients_to_remove.tolist()
        if self.unlearning_samples:
            unlearning_set["unlearning_samples"] = self.samples_to_remove.tolist()
        if self.unlearning_features:
            unlearning_set["unlearning_features"] = self.features_to_remove.tolist()
        if self.unlearning_specific_information:
            unlearning_set["unlearning_information"] = [self.information_to_remove]
        if self.unlearning_classes:
            unlearning_set["unlearning_classes"] = self.classes_to_remove.tolist()
        torch.save(unlearning_set, self.dir_path + 'unlearning_set.pkl')

    def load4stream_unlearning(self):
        self.overall_pred = torch.load(self.dir_path + 'overall_pred.pkl')
        unlearning_set = torch.load(self.dir_path + 'unlearning_set.pkl')
        assert unlearning_set["online_clients"] == self.online_clients, "online_clients not match"
        if self.unlearning_clients:
            self.clients_to_remove = np.array(self.clients_to_remove.tolist() + unlearning_set["unlearning_clients"])
        elif len(unlearning_set["unlearning_clients"]) > 0:
            self.clients_to_remove = np.array(unlearning_set["unlearning_clients"])
            self.unlearning_clients = True
        if self.unlearning_samples:
            self.samples_to_remove = np.array(self.samples_to_remove.tolist() + unlearning_set["unlearning_samples"])
        elif len(unlearning_set["unlearning_samples"]) > 0:
            self.samples_to_remove = np.array(unlearning_set["unlearning_samples"])
            self.unlearning_samples = True
        if self.unlearning_features:
            self.features_to_remove = np.array(self.features_to_remove.tolist() + unlearning_set["unlearning_features"])
        elif len(unlearning_set["unlearning_features"]) > 0:
            self.features_to_remove = np.array(unlearning_set["unlearning_features"])
            self.unlearning_features = True
        if self.unlearning_specific_information and len(unlearning_set["unlearning_information"]) > 0:
            load_information_to_remove = unlearning_set["unlearning_information"][0]
            self.information_to_remove[0] = np.concatenate((self.information_to_remove[0], load_information_to_remove[0]))
            self.information_to_remove[1] = np.concatenate((self.information_to_remove[1], load_information_to_remove[1]))
        elif len(unlearning_set["unlearning_information"]) > 0:
            self.information_to_remove = unlearning_set["unlearning_information"][0]
            self.unlearning_specific_information = True

        trained_model = torch.load(self.dir_path + 'Unlearning_model.pkl')
        for ik in range(self.k):
            self.parties[ik].local_model.load_state_dict(trained_model["model"][ik].state_dict())
            self.parties[ik].prepare_data_loader(batch_size=self.batch_size)
        self.parties[self.k-1].global_model.load_state_dict(trained_model["global_model"].state_dict())
        self.communication_cost = trained_model["Total communication cost"]
        self.total_time = trained_model["Total time"]
        
            


        

        


