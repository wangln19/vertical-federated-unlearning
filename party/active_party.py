import json
import sys, os

sys.path.append(os.pardir)
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from loguru import logger
from party.party import Party
from party.llm_party import Party as Party_LLM
from utils.basic_functions import cross_entropy_for_onehot, tf_distance_cov_cor, pairwise_dist
from utils import timer
from dataset.party_dataset import ActiveDataset
from framework.client.DistributedCommunication import convert_pred_to_msg, convert_msg_to_pred, convert_tensor_to_msg, convert_msg_to_tensor

from config import vfl_basic_config


class ActiveParty_LLM(Party_LLM):
    def __init__(self, args, index, need_data=True, need_model=True):
        print(f'==== initialize ActiveParty_LLM : party {index}======')
        logger.debug(f'running on cuda{os.getenv("CUDA_VISIBLE_DEVICES").split(",")[torch.cuda.current_device()]}')

        super().__init__(args, index, need_data=need_data, need_model=need_model)
        self.name = "server#" + str(index + 1)
        self.criterion = cross_entropy_for_onehot
        # self.encoder = args.encoder

        self.train_index = None  # args.idx_train
        self.test_index = None  # args.idx_test

        self.gt_one_hot_label = None

        self.pred_received = []
        for _ in range(args.k):
            self.pred_received.append([])

        self.global_output = None  # transmitted to passive party
        self.global_loss = None  # transmitted from passive party
        self.global_gradients = None  # transmitted from passive party

        self.weights_grad_a = None

        self.encoder_hidden_states = None
        self.encoder_attention_mask = None

    # def prepare_data_loader(self, **kwargs):
    #     super().prepare_data_loader(self.args.batch_size, self.args.need_auxiliary)


    def prepare_data(self, args, index):
        print('Active Party has no data, only global model')

    def receive_pred(self, pred, giver_index):
        self.pred_received[giver_index] = pred

    def receive_attention_mask(self, attention_mask):
        self.local_batch_attention_mask = attention_mask

    def receive_token_type_ids(self, token_type_ids):
        self.local_batch_token_type_ids = token_type_ids


    def _do_aggregate_remote(self, pred_list):
        new_dict = convert_msg_to_pred(pred_list)
        if self.args.model_type == 'XLNet':
            new_dict['output_g'] = None
        result = self.aggregate([new_dict])

        if self.args.task_type == 'CausalLM':  # self.passive_pred_list[0] = [intermediate, attention_mask]
            if self.args.model_type == 'qwen2':
                return convert_pred_to_msg(result)
            return convert_tensor_to_msg(result.logits)
        elif self.args.task_type == 'SequenceClassification':  # self.passive_pred_list[0] = [intermediate, ,sequence_lengths, attention_mask]
            return {
                "requires_grad": result.logits.requires_grad,
                "logits": result.logits.tolist()
            }
        elif self.args.task_type == 'QuestionAnswering':  # self.passive_pred_list[0] = [intermediate, attention_mask]
            return {
                "requires_grad": True,
                "start_logits": result.start_logits.tolist(),
                "end_logits": result.end_logits.tolist(),
            }
        elif self.args.task_type == 'DevLLMInference':
            return convert_pred_to_msg(result)
        else:
            assert 1 > 2, 'Task type no supported'

    def aggregate_remote(self, pred_list):
        return self._do_aggregate_remote(pred_list)

    def save_pretrained_remote(self, data):
        model_index = data['model_index']
        model_id = data['model_id']
        self.save_pretrained(model_index, model_id)

    @timer()
    def aggregate(self, pred_list, use_cache=False, test=False):
        # print(' == Active Aggregate == ')

        self.passive_pred_list = pred_list
        self.passive_pred_list[0].update({'use_cache':use_cache})
        self._tensor_to_device(self.passive_pred_list[0],self.device)
        self.global_output = self.forward(model_index=1,**self.passive_pred_list[0])  # use_cache = use_cache,return_dict=True
        if not isinstance(self.global_output,dict):
            self.global_output = self.global_output.prepare_for_forward()
        return self._detach_tensor(self.global_output)

    def receive_loss_and_gradients_remote(self, data):
        gradients = convert_msg_to_tensor(data)
        gradients = gradients.to(self.device)
        self.receive_loss_and_gradients(gradients)

    def receive_loss_and_gradients(self, gradients):
        # self.global_loss = loss
        self.global_gradients = gradients
        # print('Active Party receive self.global_gradients:')
        # print(self.global_gradients)

    def global_LR_decay(self, i_epoch):
        if self.global_model_optimizer != None:
            eta_0 = self.args.main_lr
            eta_t = eta_0 / (np.sqrt(int(i_epoch) + 1))
            for param_group in self.global_model_optimizer.param_groups:
                param_group['lr'] = eta_t
        elif self.lr_schedulers.get(1):
            self.lr_schedulers[1].step()

    def cal_passive_local_gradient(self, ik, remote=True):
        if remote:
            ik = int(ik)
        if self.args.task_type == 'QuestionAnswering':
            passive_local_gradient = \
            torch.autograd.grad(self.global_output.start_logits + self.global_output.end_logits,
                                self.passive_pred_list[ik]['inputs_embeds'], \
                                grad_outputs=self.global_gradients, retain_graph=True)[0].detach().clone()
        else:
            passive_local_gradient = \
                torch.autograd.grad(self.output_tensors[1], self.passive_pred_list[ik]['inputs_embeds'], \
                                    grad_outputs=self.global_gradients, retain_graph=True)[0].detach().clone()
        if remote:
            return passive_local_gradient.tolist()
        return passive_local_gradient

    def global_backward(self):
        # print('=== Active Global Backward ===')

        if self.global_model_optimizer != None:
            if self.args.model_architect == 'TQA': #self.args.task_type == 'QuestionAnswering':
                # update global model
                self.global_model_optimizer.zero_grad()

                # trainable layer parameters
                parameters = []

                # load grads into parameters
                weights_grad_a_start = torch.autograd.grad(self.global_output.start_logits,
                                                           self.global_model.head_layer.parameters(),
                                                           grad_outputs=self.global_gradients, retain_graph=True)
                weights_grad_a_end = torch.autograd.grad(self.global_output.end_logits,
                                                         self.global_model.head_layer.parameters(),
                                                         grad_outputs=self.global_gradients, retain_graph=True)

                self.weights_grad_a = []
                for _i in range(len(weights_grad_a_start)):
                    self.weights_grad_a.append(weights_grad_a_start[_i] + weights_grad_a_end[_i])
                self.weights_grad_a = tuple(self.weights_grad_a)

                for w, g in zip(self.global_model.head_layer.parameters(), self.weights_grad_a):
                    if w.requires_grad:
                        w.grad = g.detach()

                self.global_model_optimizer.step()

            else:
                # update global model
                try:
                    # todo: here should update all trainable params
                    self.global_model_optimizer.zero_grad()
                    self.global_gradients = self.global_gradients.to(self.global_output.logits.device)
                    weights_grad_a = torch.autograd.grad(self.global_output.logits,
                                                         self.global_model.head_layer.parameters(), \
                                                         grad_outputs=self.global_gradients, retain_graph=True)
                    self.weights_grad_a = weights_grad_a

                    for w, g in zip(self.global_model.head_layer.parameters(), weights_grad_a):
                        if w.requires_grad:
                            w.grad = g.detach()

                    self.global_model_optimizer.step()

                except Exception as e:
                    logger.debug(f"active party step optimizer 1")
                    self.global_model_optimizer.zero_grad()
                    self.global_gradients=self.global_gradients.to(self.output_tensors[1].device)
                    self.output_tensors[1].backward(gradient=self.global_gradients, retain_graph=True)
                    self.global_model_optimizer.step()
                    self.global_model_optimizer.zero_grad()

    @property
    def device(self):
        return self.models[1].device

class ActiveParty(Party):
    def __init__(self, args, index):
        super().__init__(args, index)
        self.criterion = cross_entropy_for_onehot
        self.encoder = args.encoder
        # print(f"in active party, encoder=None? {self.encoder==None}, {self.encoder}")
        self.train_index = args.idx_train
        self.test_index = args.idx_test

        self.gt_one_hot_label = None

        self.pred_received = []
        for _ in range(args.k):
            self.pred_received.append([])

        self.global_pred = None
        self.global_loss = None
        self.save_distribute_percent = args.save_distribute_percent

    def calculate_loss(self, overall_pred, test=False):
        if self.train_index != None:  # for graph data
            if test == False:
                loss = self.criterion(self.global_pred[self.train_index], self.gt_one_hot_label[self.train_index])
            else:
                loss = self.criterion(self.global_pred[self.test_index], self.gt_one_hot_label[self.test_index])
        else:
            loss = self.criterion(overall_pred, self.gt_one_hot_label)
        self.global_loss = loss
        
        return overall_pred, loss

    def unlearning_calculate_gradient(self, overall_pred, online_clients, test=False):
        # ==========not change the overall pred, only adjust for the torch.autograd.grad==============
        pred_list = []
        pred_list_clone = []
        for ik in online_clients:

            pred_list.append(self.pred_received[ik])
            pred_list_clone.append(self.pred_received[ik].detach().clone())

        agged_pred = self.global_model(pred_list)
        agged_pred_clone = self.global_model(pred_list_clone)
        # check if there is nan or inf values
        if torch.isnan(overall_pred).any() or torch.isinf(overall_pred).any():
            raise ValueError("overall_pred contains nan or inf values")
        overall_pred += agged_pred
        overall_pred -= agged_pred_clone
        # ==========not change the overall pred, only adjust for the torch.autograd.grad==============

        self.global_pred = overall_pred
        _overall_pred, loss = self.calculate_loss(overall_pred, test=test)

        pred_gradients_list = []
        pred_gradients_list_clone = []
        for ik in online_clients:
            # 计算梯度
            gradient = torch.autograd.grad(loss, self.pred_received[ik], retain_graph=True, create_graph=True)
            pred_gradients_list.append(gradient)
            # print('pred_received:', self.pred_received[ik])
            # print('gradient:', gradient, gradient[0].shape)
            # print(f"in gradient_calculation, party#{ik}, loss={loss}, pred_gradeints={pred_gradients_list[-1]}")
            pred_gradients_list_clone.append(gradient[0].detach().clone())

        return pred_gradients_list_clone

    def prepare_data(self, args, index, load_deleted=False):
        super().prepare_data(args, index, load_deleted)
        self.train_dst = ActiveDataset(self.train_data, self.train_label)
        self.test_dst = ActiveDataset(self.test_data, self.test_label)
        if self.args.need_auxiliary == 1:
            self.aux_dst = ActiveDataset(self.aux_data, self.aux_label)
            # self.aux_loader = DataLoader(self.aux_dst, batch_size=batch_size,shuffle=True)

    def update_local_pred(self, pred):
        self.pred_received[self.args.k - 1] = pred

    def receive_pred(self, pred, giver_index):
        self.pred_received[giver_index] = pred

    def save_received_pred(self):
        if self.save_distribute_percent:
            pred_received_clone = self.pred_received.copy()
            pred_received_clone = [pred.detach().cpu().numpy() for pred in pred_received_clone]
            # print(pred_received_clone.shape) # (k, batch_size, num_classes)
            return pred_received_clone
        else:
            print('save_distribute_percent is False')
            raise NotImplementedError

    def aggregate(self, pred_list, gt_one_hot_label, test=False):
        if self.args.dataset == 'cora' and self.args.apply_trainable_layer == 1:
            pred = self.global_model(pred_list, self.local_batch_data)
        else:
            pred = self.global_model(pred_list)

        if self.train_index != None:  # for graph data
            if test == False:
                loss = self.criterion(pred[self.train_index], gt_one_hot_label[self.train_index])
            else:
                loss = self.criterion(pred[self.test_index], gt_one_hot_label[self.test_index])
        else:
            loss = self.criterion(pred, gt_one_hot_label)

        # ########## for active mid model loss (start) ##########
        if self.args.apply_mid == True and (self.index in self.args.defense_configs['party']):
            # print(f"in active party mid, label={gt_one_hot_label}, global_model.mid_loss_list={self.global_model.mid_loss_list}")
            assert len(pred_list) - 1 == len(self.global_model.mid_loss_list)
            for mid_loss in self.global_model.mid_loss_list:
                loss = loss + mid_loss
            self.global_model.mid_loss_list = [torch.empty((1, 1)).to(self.args.device) for _ in
                                               range(len(self.global_model.mid_loss_list))]
        # ########## for active mid model loss (end) ##########
        elif self.args.apply_dcor == True and (self.index in self.args.defense_configs['party']):
            # print('dcor active defense')
            self.distance_correlation_lambda = self.args.defense_configs['lambda']
            # loss = criterion(pred, gt_one_hot_label) + self.distance_correlation_lambda * torch.mean(torch.cdist(pred_a, gt_one_hot_label, p=2))
            for ik in range(self.args.k - 1):
                loss += self.distance_correlation_lambda * torch.log(
                    tf_distance_cov_cor(pred_list[ik], gt_one_hot_label))  # passive party's loss
        return pred, loss

    def gradient_calculation(self, pred_list, loss):
        pred_gradients_list = []
        pred_gradients_list_clone = []
        for ik in range(self.args.k):
            pred_gradients_list.append(torch.autograd.grad(loss, pred_list[ik], retain_graph=True, create_graph=True))
            # print(f"in gradient_calculation, party#{ik}, loss={loss}, pred_gradeints={pred_gradients_list[-1]}")
            pred_gradients_list_clone.append(pred_gradients_list[ik][0].detach().clone())
        # self.global_backward(pred, loss)
        return pred_gradients_list, pred_gradients_list_clone

    def give_gradient(self):
        pred_list = self.pred_received

        if self.gt_one_hot_label == None:
            print('give gradient:self.gt_one_hot_label == None')
            assert 1 > 2

        self.global_pred, self.global_loss = self.aggregate(pred_list, self.gt_one_hot_label)
        pred_gradients_list, pred_gradients_list_clone = self.gradient_calculation(pred_list, self.global_loss)
        # self.local_gradient = pred_gradients_list_clone[self.args.k-1] # update local gradient

        if self.args.defense_name == "GradPerturb":
            self.calculate_gradient_each_class(self.global_pred, pred_list)

        return pred_gradients_list_clone

    def update_local_gradient(self, gradient):
        self.local_gradient = gradient

    def global_LR_decay(self, i_epoch):
        if self.global_model_optimizer != None:
            eta_0 = self.args.main_lr
            eta_t = eta_0 / (np.sqrt(i_epoch + 1))
            for param_group in self.global_model_optimizer.param_groups:
                param_group['lr'] = eta_t

    def global_backward(self):
        # if self.apply_R2S:
        #     print('Not implemented for trainabel top model')

        if self.global_model_optimizer != None:
            # active party with trainable global layer
            _gradients = torch.autograd.grad(self.global_loss, self.global_pred, retain_graph=True)
            _gradients_clone = _gradients[0].detach().clone()

            # if self.args.apply_mid == False and self.args.apply_trainable_layer == False:
            #     return # no need to update

            # update global model
            self.global_model_optimizer.zero_grad()
            parameters = []
            if (self.args.apply_mid == True) and (self.index in self.args.defense_configs['party']):
                # mid parameters
                for mid_model in self.global_model.mid_model_list:
                    parameters += list(mid_model.parameters())
                # trainable layer parameters
                if self.args.apply_trainable_layer == True:
                    parameters += list(self.global_model.global_model.parameters())

                # load grads into parameters
                weights_grad_a = torch.autograd.grad(self.global_pred, parameters, grad_outputs=_gradients_clone,
                                                     retain_graph=True)
                for w, g in zip(parameters, weights_grad_a):
                    if w.requires_grad:
                        w.grad = g.detach()

            else:
                # trainable layer parameters
                if self.args.apply_trainable_layer == True:
                    # load grads into parameters
                    weights_grad_a = torch.autograd.grad(self.global_pred, self.global_model.parameters(),
                                                         grad_outputs=_gradients_clone, retain_graph=True)
                    for w, g in zip(self.global_model.parameters(), weights_grad_a):
                        if w.requires_grad:
                            w.grad = g.detach()
                # non-trainabel layer: no need to update
            self.global_model_optimizer.step()

    def calculate_gradient_each_class(self, global_pred, local_pred_list, test=False):
        # print(f"global_pred.shape={global_pred.size()}") # (batch_size, num_classes)
        self.gradient_each_class = [[] for _ in range(global_pred.size(1))]
        one_hot_label = torch.zeros(global_pred.size()).to(global_pred.device)
        for ic in range(global_pred.size(1)):
            one_hot_label *= 0.0
            one_hot_label[:, ic] += 1.0
            if self.train_index != None:  # for graph data
                if test == False:
                    loss = self.criterion(global_pred[self.train_index], one_hot_label[self.train_index])
                else:
                    loss = self.criterion(global_pred[self.test_index], one_hot_label[self.test_index])
            else:
                loss = self.criterion(global_pred, one_hot_label)
            for ik in range(self.args.k):
                self.gradient_each_class[ic].append(
                    torch.autograd.grad(loss, local_pred_list[ik], retain_graph=True, create_graph=True))
        # end of calculate_gradient_each_class, return nothing
