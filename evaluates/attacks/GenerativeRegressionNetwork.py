import sys, os

sys.path.append(os.pardir)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
import numpy as np
import copy
import pickle
import matplotlib.pyplot as plt
import itertools

from evaluates.attacks.attacker import Attacker
from models.global_models import *
from utils.basic_functions import cross_entropy_for_onehot, append_exp_res
from dataset.party_dataset import PassiveDataset
from dataset.party_dataset import ActiveDataset

##### add DP noise without torch.no_grad()
from evaluates.defenses.defense_functions import LaplaceDP_for_pred_grn, GaussianDP_for_pred_grn


def cal_ssim(im1, im2):
    assert len(im1.shape) == 2 and len(im2.shape) == 2
    assert im1.shape == im2.shape
    mu1 = im1.mean()
    mu2 = im2.mean()
    sigma1 = np.sqrt(((im1 - mu1) ** 2).mean())
    sigma2 = np.sqrt(((im2 - mu2) ** 2).mean())
    sigma12 = ((im1 - mu1) * (im2 - mu2)).mean()
    k1, k2, L = 0.01, 0.03, 255
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    C3 = C2 / 2
    l12 = (2 * mu1 * mu2 + C1) / (mu1 ** 2 + mu2 ** 2 + C1)
    c12 = (2 * sigma1 * sigma2 + C2) / (sigma1 ** 2 + sigma2 ** 2 + C2)
    s12 = (sigma12 + C3) / (sigma1 * sigma2 + C3)
    ssim = l12 * c12 * s12
    return ssim


def label_to_one_hot(target, num_classes=10):
    # print('label_to_one_hot:', target, type(target))
    try:
        _ = target.size()[1]
        # print("use target itself", target.size())
        onehot_target = target.type(torch.float32)
    except:
        target = torch.unsqueeze(target, 1)
        # print("use unsqueezed target", target.size())
        onehot_target = torch.zeros(target.size(0), num_classes)
        onehot_target.scatter_(1, target, 1)
    return onehot_target


# class Generator(nn.Module):
#     def __init__(self, latent_dim, target_dim):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(latent_dim, 600),
#             nn.LayerNorm(600),
#             nn.ReLU(),

#             nn.Linear(600, 200),
#             nn.LayerNorm(200),
#             nn.ReLU(),

#             nn.Linear(200, 100),
#             nn.LayerNorm(100),
#             nn.ReLU(),

#             nn.Linear(100, target_dim),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         x = torch.tensor(x, dtype=torch.float32)
#         return self.net(x)
    

class Generator(nn.Module):
    def __init__(self, latent_dim, target_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(latent_dim, 20),
            nn.LayerNorm(20),
            nn.ReLU(),

            nn.Linear(20, 10),
            nn.LayerNorm(10),
            nn.ReLU(),

            nn.Linear(10, target_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        return self.net(x)


class GenerativeRegressionNetwork(Attacker):
    def __init__(self, top_vfl, args):
        super().__init__(args)
        self.args = args
        self.vfl_info = top_vfl.final_state
        self.party_info = top_vfl.parties
        # prepare parameters
        self.device = args.device
        self.num_classes = args.num_classes
        self.label_size = args.num_classes
        self.k = args.k
        self.batch_size = args.batch_size

        # attack configs
        self.party = args.attack_configs['party']  # parties that launch attacks
        self.lr = args.attack_configs['lr']
        self.epochs = args.attack_configs['epochs']
        # self.data_number = args.attack_configs['data_number'] #2048
        self.grn_batch_size = args.attack_configs['batch_size']  # 64
        self.unknownVarLambda = 0.25  # by default

        self.criterion = cross_entropy_for_onehot

    def set_seed(self, seed=0):
        # random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    def MSE_PSNR(self, batch_real_image, batch_dummy_image):
        '''
        compute MSE and PSNR
        :param batch_real_image:
        :param batch_dummy_image:
        :return:
        '''
        # print(batch_real_image.size(),batch_dummy_image.size())
        batch_real_image = batch_real_image.reshape(batch_dummy_image.size())
        mse = torch.mean((batch_real_image - batch_dummy_image) ** 2)
        psnr = 20 * torch.log10(1 / torch.sqrt(mse))
        return mse.cpu().numpy(), psnr.cpu().numpy()

    def slt_tgt_loss(self, mse, rand_mse, start, end):
        mse, rand_mse = mse.cpu().numpy(), rand_mse.cpu().numpy()
        if self.args.attack_specific_information:
            mask = (self.args.information_to_remove[0] >= start) & (self.args.information_to_remove[0] < end)
            filtered_indices = (self.args.information_to_remove[0][mask] - start, self.args.information_to_remove[1][mask])
            mse = mse[filtered_indices]
            rand_mse = rand_mse[filtered_indices]
        elif self.args.attack_features:
            mse = mse[:, self.args.features_to_remove]
            rand_mse = rand_mse[:, self.args.features_to_remove]
        return mse, rand_mse  

    def slt_ass_data(self, data):
        data = np.array(data)
        if self.args.attack_specific_information:
            col_to_remove = np.unique(self.args.information_to_remove[1])
        elif self.args.attack_features:
            col_to_remove = self.args.features_to_remove
        else:
            return data
        # rand_data = torch.randn([data.shape[0], len(col_to_remove)]).to(self.device)
        rand_data = np.random.rand(data.shape[0], len(col_to_remove))
        data[:, col_to_remove] = rand_data
        return data

    
    def attack(self):
        # self.set_seed(123)
        print_every = 1
        for ik in self.party:  # attacker party #ik
            assert ik == self.k-1, 'Only Active party launch feature inference attack'
            index = ik
            # collect necessary information
            net_b = [None for _ in range(self.k - 1)]
            for i in range(self.k-1):
                # net_b[i] = self.vfl_info['model'][i].to(self.device)
                net_b[i] = self.party_info[i].local_model.to(self.device)
                net_b[i].eval()
            # net_a = self.vfl_info['model'][self.k-1].to(self.device)  # Active
            net_a = self.party_info[self.k-1].local_model.to(self.device)
            global_model = self.vfl_info['global_model'].to(self.device)
            global_model.eval()
            net_a.eval()

            # Train with Aux Dataset
            # data_number = self.data_number
            batch_size = self.grn_batch_size

            train_data_b = [None for _ in range(self.k - 1)]
            train_dst_b = [None for _ in range(self.k - 1)]
            train_loader_b = [None for _ in range(self.k - 1)]
            test_data_b = [None for _ in range(self.k - 1)]
            test_dst_b = [None for _ in range(self.k - 1)]
            test_loader_b = [None for _ in range(self.k - 1)]
            # inverse the train and test data
            train_data_a = self.vfl_info["test_data"][self.k - 1]
            train_label = self.vfl_info["test_label"][self.k - 1]
            train_dst_a = ActiveDataset(train_data_a, train_label)
            train_loader_a = DataLoader(train_dst_a, batch_size=batch_size)
            test_data_a = self.vfl_info['train_data'][self.k - 1]
            tset_label = self.vfl_info['train_label'][self.k - 1]
            test_dst_a = ActiveDataset(test_data_a, tset_label)
            test_loader_a = DataLoader(test_dst_a, batch_size=batch_size)
            test_num = len(test_data_a)
            for i in range(self.k - 1):
                train_data_b[i] = self.vfl_info["test_data"][i]
                train_dst_b[i] = PassiveDataset(train_data_b[i])
                train_loader_b[i] = DataLoader(train_dst_b[i], batch_size=batch_size)
                test_data_b[i] = self.vfl_info['train_data'][i]
                test_dst_b[i] = PassiveDataset(test_data_b[i])
                test_loader_b[i] = DataLoader(test_dst_b[i], batch_size=batch_size)
            train_loader_list = train_loader_b + [train_loader_a]
            test_loader_list = test_loader_b + [test_loader_a]

            if self.args.dataset in ['credit']:
                dim_a = test_data_a.size()[1]
                dim_b = 0
                for i in range(self.k - 1):
                    dim_b += test_data_b[i].size()[1]
                dim_tgt = test_data_b[0].size()[1]
            self.netG = Generator(dim_a + dim_b, dim_tgt)
            self.netG = self.netG.to(self.device)
            self.optimizerG = torch.optim.Adam(self.netG.parameters(), lr=self.lr)
            criterion = nn.MSELoss()

            # mark = 0
            # for name, param in self.netG.named_parameters():
            #     if mark == 0:
            #         print(name, param)
            #         mark = mark + 1

            early_stop = 0
            min_mse = 1000000
            print('========= Feature Inference Training ========')
            for i_epoch in range(self.epochs):
                self.netG.train()
                for parties_data in zip(*train_loader_list):

                    batch_data_b = [None for _ in range(self.k - 1)]
                    for i in range(self.k - 1):
                        batch_data_b[i] = parties_data[i][0]
                    batch_data_a = parties_data[self.k - 1][0]

                    self.optimizerG.zero_grad()
                    
                    # generate "fake inputs"
                    noise_data_b = torch.randn(batch_data_b[0].size()).to(self.device)
                    # noise_data_b = torch.tensor(self.slt_ass_data(batch_data_b[0].detach().cpu().numpy())).to(self.device)
                    if self.args.dataset in ['credit']:
                        for i in range(self.k - 1):
                            if i == 0:
                                generated_data_b_input = noise_data_b
                            else:
                                generated_data_b_input = torch.cat((generated_data_b_input, batch_data_b[i]), dim=1)
                        generated_data_b_input = torch.cat((generated_data_b_input, batch_data_a), dim=1)
                        generated_data_b = self.netG(generated_data_b_input)
                    generated_data_b = generated_data_b.reshape(batch_data_b[0].size())
                    # with torch.no_grad():
                    #     ass_data = self.slt_ass_data(batch_data_b[0].detach().cpu().numpy())
                    #     generated_data_b = generated_data_b + torch.tensor(ass_data).to(self.device)
                    # compute logits of generated/real data
                    pred_a = net_a(batch_data_a)
                    dummy_pred_b = net_b[0](generated_data_b)
                    pred_b = [None for _ in range(self.k - 1)]
                    for i in range(self.k - 1):
                        pred_b[i] = net_b[i](batch_data_b[i])

                    ####### DP Defense On FR ########
                    if self.args.apply_dp == True:
                        if 'laplace' in self.args.defense_name.casefold():
                            pred_b = LaplaceDP_for_pred_grn(self.args, pred_b)
                            dummy_pred_b = LaplaceDP_for_pred_grn(self.args, dummy_pred_b)
                        elif 'gaussian' in self.args.defense_name.casefold():
                            # print("before",pred_b.shape,pred_b)
                            pred_b = GaussianDP_for_pred_grn(self.args, pred_b)
                            dummy_pred_b = GaussianDP_for_pred_grn(self.args, dummy_pred_b)
                            # print("after",pred_b.shape,pred_b)
                            # assert 1>2
                    ####### DP Defense On FR ########

                    # aggregate logits of clients
                    real_pred = global_model(pred_b + [pred_a])
                    dummy_pred = global_model([dummy_pred_b] + pred_b[1:] + [pred_a])

                    # print('dummy_pred_b:',dummy_pred_b.requires_grad)
                    # print('dummy_pred:',dummy_pred.requires_grad)

                    unknown_var_loss = 0.0
                    for i in range(generated_data_b.size(0)):
                        unknown_var_loss = unknown_var_loss + (generated_data_b[i].var())  # var() unknown
                    # if self.args.dataset == 'nuswide' and torch.sum(dummy_pred,dim=-1)[0] != 1.0:
                    #     loss = (((F.softmax(dummy_pred,dim=-1) - F.softmax(real_pred,dim=-1))**2).sum() \
                    #     + self.unknownVarLambda * unknown_var_loss * 0.25)
                    # else:
                    #     loss = (((dummy_pred - real_pred)**2).sum() \
                    #     + self.unknownVarLambda * unknown_var_loss * 0.25)
                    loss = (((dummy_pred - real_pred) ** 2).sum() \
                            + self.unknownVarLambda * unknown_var_loss * 0.25)
                    # if self.args.dataset == 'nuswide':
                    #     # print(f"[debug] (generated_data_b > 0.5).float().shape={(generated_data_b > 0.5).float().shape}, batch_data_b.shape={batch_data_b.shape}")
                    #     train_mse = criterion((generated_data_b > 0.5).float(), batch_data_b)
                    # else:
                    #     train_mse = criterion(generated_data_b, batch_data_b)
                    train_mse = criterion(generated_data_b, batch_data_b[0])
                    # loss = (((dummy_pred - real_pred) ** 2).sum() \
                    #         + self.unknownVarLambda * unknown_var_loss * 0.25 + train_mse)
                    # print(f'[debug] see pred_match_loss: {((F.softmax(dummy_pred,dim=-1) - F.softmax(real_pred,dim=-1))**2).sum()}')
                    # print(f'[debug] see var_loss: {unknown_var_loss}')
                    # print(f'[debug] see gradients of pred_match_loss: {torch.autograd.grad(((F.softmax(dummy_pred,dim=-1) - F.softmax(real_pred,dim=-1))**2).sum(), self.netG.parameters(), retain_graph=True)}')
                    # print(f'[debug] see gradients of var_loss: {torch.autograd.grad(unknown_var_loss, self.netG.parameters(), retain_graph=True)}')
                    # print(f'[debug] see fake data_b={generated_data_b}')
                    # print(f'[debug] see real data_b={batch_data_b}')
                    loss.backward()

                    # # check if gradient is not None
                    # mark = 0
                    # for name, param in self.netG.named_parameters():
                    #     if mark == 0:
                    #         print('Check grad:',name, param.grad)
                    #         mark = mark + 1

                    self.optimizerG.step()

                    ####### Test Performance of Generator #######
                if (i_epoch + 1) % print_every == 0:
                    self.netG.eval()
                    # MSE = []
                    # PSNR = []
                    mse_list, rand_mse_list = [], []
                    start = 0
                    end = batch_size if batch_size < test_num else test_num
                    for parties_data in zip(*test_loader_list):
                        batch_data_b = [None for _ in range(self.k - 1)]
                        for i in range(self.k - 1):
                            batch_data_b[i] = parties_data[i][0]
                        batch_data_a = parties_data[self.k - 1][0]
                        with torch.no_grad():
                            noise_data_b = torch.randn(batch_data_b[0].size()).to(self.device)
                            if self.args.dataset in ['credit']:
                                for i in range(self.k - 1):
                                    if i == 0:
                                        generated_data_b_input = noise_data_b
                                    else:
                                        generated_data_b_input = torch.cat((generated_data_b_input, batch_data_b[i]), dim=1)
                                generated_data_b_input = torch.cat((generated_data_b_input, batch_data_a), dim=1)
                                generated_data_b = self.netG(generated_data_b_input)
                            origin_data = batch_data_b[0].reshape(generated_data_b.size()).to(self.device)
                            noise_data = noise_data_b.reshape(generated_data_b.size()).to(self.device)
                            mse_loss = nn.MSELoss(reduction='none')
                            mse = mse_loss(generated_data_b, origin_data)
                            rand_mse = mse_loss(noise_data, origin_data)
                            mse, rand_mse = self.slt_tgt_loss(mse, rand_mse, start, end)
                            # slt_generated_data_b, slt_org_data = self.slt_tgt_loss(generated_data_b, origin_data, start, end)
                            # print(f'[debug] see slt_generated_data_b={slt_generated_data_b}')
                            # print(f'[debug] see slt_org_data={slt_org_data}')
                            start = end
                            end = end + batch_size if end + batch_size < test_num else test_num
                            mse_list += mse.flatten().tolist()
                            rand_mse_list += rand_mse.flatten().tolist()

                    # if i_epoch == self.epochs - 1:
                    #     generated_data_b = generated_data_b.reshape(batch_data_b[0].size())
                    rand_mse = np.mean(rand_mse_list)
                    mse = np.mean(mse_list)
                    print('Epoch {} \t train_loss:{} train_mse:{:.3f} mse:{:.3f} rand_mse:{:.3f}'.format(
                        i_epoch, loss.item(), train_mse, mse, rand_mse))
                
                if mse > min_mse:
                    early_stop += 1
                if early_stop >= 10:
                    break
                min_mse = min(min_mse, mse)
                
            # mark = 0
            # print('Final Model')
            # for name, param in self.netG.named_parameters():
            #     if mark == 0:
            #         print(name, param)
            #         mark = mark + 1

            ####### Clean ######
            del (self.netG)
            del (train_dst_a)
            del (train_loader_a)
            del (train_dst_b)
            del (train_loader_b)
            del (train_loader_list)

            print(f"GRN, if self.args.apply_defense={self.args.apply_defense}")
            print(f'batch_size=%d,class_num=%d,party_index=%d,mse=%lf' % (self.batch_size, self.label_size, index, mse))

        print("returning from GRN")
        return rand_mse, min_mse


# class GenerativeRegressionNetwork(Attacker):
#     def __init__(self, top_vfl, args):
#         super().__init__(args)
#         self.args = args
#         self.vfl_info = top_vfl.final_state
#         # prepare parameters
#         self.device = args.device
#         self.num_classes = args.num_classes
#         self.label_size = args.num_classes
#         self.k = args.k
#         self.batch_size = args.batch_size

#         # attack configs
#         self.party = args.attack_configs['party']  # parties that launch attacks
#         self.lr = args.attack_configs['lr']
#         self.epochs = args.attack_configs['epochs']
#         # self.data_number = args.attack_configs['data_number'] #2048
#         self.grn_batch_size = args.attack_configs['batch_size']  # 64
#         self.unknownVarLambda = 0.25  # by default

#         self.criterion = cross_entropy_for_onehot

#     def set_seed(self, seed=0):
#         # random.seed(seed)
#         os.environ['PYTHONHASHSEED'] = str(seed)
#         np.random.seed(seed)
#         torch.manual_seed(seed)
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = True

#     def MSE_PSNR(self, batch_real_image, batch_dummy_image):
#         '''
#         compute MSE and PSNR
#         :param batch_real_image:
#         :param batch_dummy_image:
#         :return:
#         '''
#         # print(batch_real_image.size(),batch_dummy_image.size())
#         batch_real_image = batch_real_image.reshape(batch_dummy_image.size())
#         mse = torch.mean((batch_real_image - batch_dummy_image) ** 2)
#         psnr = 20 * torch.log10(1 / torch.sqrt(mse))
#         return mse.cpu().numpy(), psnr.cpu().numpy()

#     def attack(self):
#         self.set_seed(123)
#         print_every = 1
#         for ik in self.party:  # attacker party #ik
#             assert ik == 1, 'Only Active party launch feature inference attack'
#             index = ik
#             # collect necessary information
#             net_b = self.vfl_info['final_model'][0].to(self.device)  # Passive
#             net_a = self.vfl_info['final_model'][1].to(self.device)  # Active
#             global_model = self.vfl_info['final_global_model'].to(self.device)
#             global_model.eval()
#             net_b.eval()
#             net_a.eval()

#             # Train with Aux Dataset
#             # data_number = self.data_number
#             batch_size = self.grn_batch_size

#             aux_data_a = self.vfl_info["aux_data"][1]
#             aux_data_b = self.vfl_info["aux_data"][0]
#             aux_label = self.vfl_info["aux_label"][-1]
#             aux_dst_a = ActiveDataset(aux_data_a, aux_label)
#             aux_loader_a = DataLoader(aux_dst_a, batch_size=batch_size)
#             aux_dst_b = PassiveDataset(aux_data_b)
#             aux_loader_b = DataLoader(aux_dst_b, batch_size=batch_size)
#             aux_loader_list = [aux_loader_b,aux_loader_a]

#             train_data_a = self.vfl_info["train_data"][1]
#             train_data_b = self.vfl_info["train_data"][0]
#             train_label = self.vfl_info["train_data"][-1]
#             train_dst_a = ActiveDataset(train_data_a, train_label)
#             train_loader_a = DataLoader(train_dst_a, batch_size=batch_size)
#             train_dst_b = PassiveDataset(train_data_b)
#             train_loader_b = DataLoader(train_dst_b, batch_size=batch_size)
#             train_loader_list = [train_loader_b, train_loader_a]
            
#             # Test Data
#             test_data_a = self.vfl_info['test_data'][1]  # Active Test Data
#             test_data_b = self.vfl_info['test_data'][0]  # Passive Test Data

#             # Initalize Generator
#             if self.args.dataset in ['nuswide', 'breast_cancer_diagnose', 'diabetes', 'adult_income', 'criteo']:
#                 dim_a = test_data_a.size()[1]
#                 dim_b = test_data_b.size()[1]
#             else:  # mnist cifar
#                 dim_a = test_data_a.size()[1] * test_data_a.size()[2] * test_data_a.size()[3]
#                 dim_b = test_data_b.size()[1] * test_data_b.size()[2] * test_data_b.size()[3]
#             self.netG = Generator(dim_a + dim_b, dim_b)
#             self.netG = self.netG.to(self.device)
#             self.optimizerG = torch.optim.Adam(self.netG.parameters(), lr=self.lr)
#             criterion = nn.MSELoss()

#             # mark = 0
#             # for name, param in self.netG.named_parameters():
#             #     if mark == 0:
#             #         print(name, param)
#             #         mark = mark + 1

#             print('========= Feature Inference Training ========')
#             for i_epoch in range(self.epochs):
#                 self.netG.train()
#                 for parties_data in zip(*train_loader_list):
#                     self.gt_one_hot_label = label_to_one_hot(parties_data[self.k - 1][1], self.num_classes)
#                     self.gt_one_hot_label = self.gt_one_hot_label.to(self.device)
#                     self.parties_data = parties_data
#                     batch_data_b = parties_data[0][0]  # Passive Party data
#                     batch_data_a = parties_data[1][0]  # Active Party data

#                     self.optimizerG.zero_grad()

#                     # generate "fake inputs"
#                     noise_data_b = torch.randn(batch_data_b.size()).to(
#                         self.device)  # attack from passive side, data_b is at active side need to be generated from noise at passive side
#                     # # print('batch_data_b:',batch_data_b.size())
#                     # # print('torch.cat:',batch_data_a.size(),noise_data_b.size())
#                     # # print('cat:',torch.cat((batch_data_a,noise_data_b),dim=1).size())
#                     if self.args.dataset in ['nuswide', 'breast_cancer_diagnose', 'diabetes', 'adult_income', 'criteo']:
#                         generated_data_b = self.netG(torch.cat((batch_data_a, noise_data_b), dim=1))
#                     else:
#                         generated_data_b = self.netG(torch.cat((batch_data_a, noise_data_b), dim=2))
#                     generated_data_b = generated_data_b.reshape(batch_data_b.size())

#                     # compute logits of generated/real data
#                     pred_a = net_a(batch_data_a)
#                     pred_b = net_b(batch_data_b)
#                     dummy_pred_b = net_b(generated_data_b)

#                     ####### DP Defense On FR ########
#                     if self.args.apply_dp == True:
#                         if 'laplace' in self.args.defense_name.casefold():
#                             pred_b = LaplaceDP_for_pred_grn(self.args, pred_b)
#                             dummy_pred_b = LaplaceDP_for_pred_grn(self.args, dummy_pred_b)
#                         elif 'gaussian' in self.args.defense_name.casefold():
#                             # print("before",pred_b.shape,pred_b)
#                             pred_b = GaussianDP_for_pred_grn(self.args, pred_b)
#                             dummy_pred_b = GaussianDP_for_pred_grn(self.args, dummy_pred_b)
#                             # print("after",pred_b.shape,pred_b)
#                             # assert 1>2
#                     ####### DP Defense On FR ########

#                     # aggregate logits of clients
#                     real_pred = global_model([pred_b, pred_a])
#                     dummy_pred = global_model([dummy_pred_b, pred_a])

#                     # print('dummy_pred_b:',dummy_pred_b.requires_grad)
#                     # print('dummy_pred:',dummy_pred.requires_grad)

#                     unknown_var_loss = 0.0
#                     for i in range(generated_data_b.size(0)):
#                         unknown_var_loss = unknown_var_loss + (generated_data_b[i].var())  # var() unknown
#                     # if self.args.dataset == 'nuswide' and torch.sum(dummy_pred,dim=-1)[0] != 1.0:
#                     #     loss = (((F.softmax(dummy_pred,dim=-1) - F.softmax(real_pred,dim=-1))**2).sum() \
#                     #     + self.unknownVarLambda * unknown_var_loss * 0.25)
#                     # else:
#                     #     loss = (((dummy_pred - real_pred)**2).sum() \
#                     #     + self.unknownVarLambda * unknown_var_loss * 0.25)
#                     loss = (((dummy_pred - real_pred) ** 2).sum() \
#                             + self.unknownVarLambda * unknown_var_loss * 0.25)
#                     # if self.args.dataset == 'nuswide':
#                     #     # print(f"[debug] (generated_data_b > 0.5).float().shape={(generated_data_b > 0.5).float().shape}, batch_data_b.shape={batch_data_b.shape}")
#                     #     train_mse = criterion((generated_data_b > 0.5).float(), batch_data_b)
#                     # else:
#                     #     train_mse = criterion(generated_data_b, batch_data_b)
#                     train_mse = criterion(generated_data_b, batch_data_b[0])
#                     # print(f'[debug] see pred_match_loss: {((F.softmax(dummy_pred,dim=-1) - F.softmax(real_pred,dim=-1))**2).sum()}')
#                     # print(f'[debug] see var_loss: {unknown_var_loss}')
#                     # print(f'[debug] see gradients of pred_match_loss: {torch.autograd.grad(((F.softmax(dummy_pred,dim=-1) - F.softmax(real_pred,dim=-1))**2).sum(), self.netG.parameters(), retain_graph=True)}')
#                     # print(f'[debug] see gradients of var_loss: {torch.autograd.grad(unknown_var_loss, self.netG.parameters(), retain_graph=True)}')
#                     # print(f'[debug] see fake data_b={generated_data_b}')
#                     # print(f'[debug] see real data_b={batch_data_b}')
#                     loss.backward()

#                     # # check if gradient is not None
#                     # mark = 0
#                     # for name, param in self.netG.named_parameters():
#                     #     if mark == 0:
#                     #         print('Check grad:',name, param.grad)
#                     #         mark = mark + 1

#                     self.optimizerG.step()

#                     ####### Test Performance of Generator #######
#                 if (i_epoch + 1) % print_every == 0:
#                     self.netG.eval()
#                     # MSE = []
#                     # PSNR = []
#                     with torch.no_grad():
#                         noise_data_b = torch.randn(test_data_b.size()).to(self.device)

#                         if self.args.dataset in ['nuswide', 'breast_cancer_diagnose', 'diabetes', 'adult_income',
#                                                  'criteo']:
#                             generated_data_b = self.netG(torch.cat((test_data_a, noise_data_b), dim=1))
#                         else:
#                             generated_data_b = self.netG(torch.cat((test_data_a, noise_data_b), dim=2))

#                         origin_data = test_data_b.reshape(generated_data_b.size()).to(self.device)
#                         noise_data = noise_data_b.reshape(generated_data_b.size()).to(self.device)
#                         # if self.args.dataset == 'nuswide':
#                         #     mse = criterion((generated_data_b > 0.5).float(), origin_data)
#                         #     rand_mse = criterion((noise_data > 0.5).float(), origin_data) #1.006
#                         # else:
#                         #     mse = criterion(generated_data_b, origin_data)
#                         #     rand_mse = criterion(noise_data, origin_data) #1.006
#                         mse = criterion(generated_data_b, origin_data)
#                         rand_mse = criterion(noise_data, origin_data)  # 1.006

#                         # MSE.append(mse)
#                         # PSNR.append(psnr)

#                     if i_epoch == self.epochs - 1:
#                         generated_data_b = generated_data_b.reshape(test_data_b.size())
#                         torch.save(generated_data_b, f"./exp_result/grn/{self.args.defense_name}.pkl")

#                     print('Epoch {}% \t train_loss:{} train_mse:{:.3f} mse:{:.3f} rand_mse:{:.3f}'.format(
#                         i_epoch, loss.item(), train_mse, mse, rand_mse))

#             # mark = 0
#             # print('Final Model')
#             # for name, param in self.netG.named_parameters():
#             #     if mark == 0:
#             #         print(name, param)
#             #         mark = mark + 1

#             ####### Clean ######
#             del (self.netG)
#             del (train_dst_a)
#             del (train_loader_a)
#             del (train_dst_b)
#             del (train_loader_b)
#             del (train_loader_list)

#             print(f"GRN, if self.args.apply_defense={self.args.apply_defense}")
#             print(f'batch_size=%d,class_num=%d,party_index=%d,mse=%lf' % (self.batch_size, self.label_size, index, mse))

#         print("returning from GRN")
#         return rand_mse, mse
