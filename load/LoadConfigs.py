import sys, os

sys.path.append(os.pardir)
import math
import json
import argparse
from models.autoencoder import AutoEncoder
import numpy as np
import torch

TARGETED_BACKDOOR = ['ReplacementBackdoor', 'ASB']
UNTARGETED_BACKDOOR = ['NoisyLabel', 'MissingFeature', 'NoisySample']
LABEL_INFERENCE = ['BatchLabelReconstruction', 'DirectLabelScoring', 'NormbasedScoring', \
                   'DirectionbasedScoring', 'PassiveModelCompletion', 'ActiveModelCompletion']
ATTRIBUTE_INFERENCE = ['AttributeInference']
FEATURE_INFERENCE = ['GenerativeRegressionNetwork', 'ResSFL']
# LLM attacks
INVERSION = ["VanillaModelInversion_WhiteBox", "VanillaModelInversion_BlackBox", "WhiteBoxInversion"]

communication_protocol_list = ['FedSGD', 'FedBCD_p', 'FedBCD_s', 'CELU', 'Quantization', 'Topk']


def load_online_clients_configs(args):
    file_path = './configs/online_clients.txt'
    with open(file_path, 'r') as f:
        online_clients = f.readlines()
    online_clients = [int(i) for i in online_clients]
    online_clients = np.array(online_clients)
    
    args.online_clients = online_clients
    return args


def load_trained_models_configs(args):
    # Load trained models
    # file_path = './configs/NoDefense.pkl'
    if args.wth_DP:
        file_path = args.exp_res_dir + f'trained_models/DP/Trained_model.pkl'
    elif args.regular_h:
        file_path = args.exp_res_dir + f'trained_models/VFULR_orginal/Trained_model.pkl'
    else:
        file_path = args.exp_res_dir + f'trained_models/orginal/Trained_model.pkl'
    trained_model, distributed_percent, overall_pred = torch.load(file_path)
    # trained_model, model_names, distributed_percent, overall_pred = torch.load(file_path)
    # for party 0,1,2,...,args.k-1
    for ik in range(args.k):
        args.parties[ik].local_model.load_state_dict(trained_model["model"][ik].state_dict())
        # args.parties[ik].local_model.load_state_dict(trained_model[ik])
    overall_pred = np.array(overall_pred)
    # distributed_percent = np.array(distributed_percent)
    args.overall_pred = overall_pred
    args.distributed_percent = torch.load('configs/historical_clients_distribution.pkl')
    # args.distributed_percent = distributed_percent
    # print('overall_pred:', overall_pred.shape, overall_pred)
    # print('distributed_percent:', distributed_percent.shape, distributed_percent)
    return args


def load_unlearning_features_configs(args):
    # file_path = './configs/reduce_features.txt'
    assert len(args.src_file_dir) > 0, 'src_file_dir is empty'
    file_path = args.exp_res_dir + args.src_file_dir + 'reduce_features.txt'
    with open(file_path, 'r') as f:
        reduce_features = f.readlines()
    reduce_features = [int(i) for i in reduce_features]
    reduce_features = np.array(reduce_features)
    # print('reduce_features:', reduce_features)
    # print('reduce_features:', np.array(reduce_features).shape)
    
    args.features_to_remove = reduce_features
    return args


def load_unlearning_samples_configs(args):
    # file_path = './configs/reduce_samples.txt'
    assert len(args.src_file_dir) > 0, 'src_file_dir is empty'
    file_path = args.exp_res_dir + args.src_file_dir + 'reduce_samples.txt'
    with open(file_path, 'r') as f:
        remove_samples = f.readlines()
    remove_samples = [int(i) for i in remove_samples]
    remove_samples = np.array(remove_samples)
    # print('remove_samples:', remove_samples)
    # print('remove_samples:', np.array(remove_samples).shape)
    args.samples_to_remove = remove_samples
    return args


def load_unlearning_information_configs(args):
    # file_path = './configs/reduce_information.txt'
    assert len(args.src_file_dir) > 0, 'src_file_dir is empty'
    file_path = args.exp_res_dir + args.src_file_dir + 'reduce_information.npy'
    reduce_information = np.load(file_path)
    # print('reduce_information:', reduce_information)
    # print('reduce_information:', np.array(reduce_information).shape)
    args.information_to_remove = reduce_information
    return args


def load_unlearning_clients_configs(args):
    file_path = './configs/reduce_clients.txt'
    with open(file_path, 'r') as f:
        remove_clients = f.readlines()
    remove_clients = [int(i) for i in remove_clients]
    remove_clients = np.array(remove_clients)
    # print('remove_clients:', remove_clients)
    # print('remove_clients:', np.array(remove_clients).shape)
    
    args.clients_to_remove = remove_clients
    return args


def load_llm_configs(config_dict):
    args = do_load_basic_configs(config_dict, argparse.Namespace())
    # args.tasks = config_dict['tasks']
    args.device = config_dict['device']
    args.gpu = config_dict['gpu']
    args.fl_type = config_dict['fl_type']
    return args


def load_basic_configs(config_file_name, args):
    config_file_path = './configs/' + config_file_name + '.json'
    config_file = open(config_file_path, "r")
    config_dict = json.load(config_file)
    args = do_load_basic_configs(config_dict, args)
    return args


def do_load_basic_configs(config_dict, args):
    # print(config_dict)
    args.passive_party_class = config_dict['passive_party_class'] if (
                'passive_party_class' in config_dict) else "PassiveParty_LLM"
    args.active_party_class = config_dict['active_party_class'] if (
                'active_party_class' in config_dict) else "ActiveParty_LLM"

    # args.main_lr, learning rate for main task
    args.main_lr = config_dict['lr'] if ('lr' in config_dict) else 0.001
    assert (args.main_lr > 0), "main learning rate should be >0"

    # args.main_epochs, iterations for main task
    args.main_epochs = config_dict['epochs'] if ('epochs' in config_dict) else 50

    # args.early_stop_threshold, early stop max epoch
    args.early_stop_threshold = config_dict['early_stop_threshold'] if ('early_stop_threshold' in config_dict) else 5

    # args.k, number of participants
    args.k = config_dict['k'] if ('k' in config_dict) else 2
    assert (args.k % 1 == 0 and args.k > 0), "k should be positive integers"

    # args.batch_size for main task
    args.batch_size = config_dict['batch_size'] if ('batch_size' in config_dict) else 2048
    args.test_batch_size = config_dict['test_batch_size'] if ('test_batch_size' in config_dict) else args.batch_size

    # Communication Protocol
    communication_protocol_dict = config_dict['communication'] if ('communication' in config_dict) else None

    args.communication_protocol = communication_protocol_dict['communication_protocol'] if (
                'communication_protocol' in communication_protocol_dict) else 'FedBCD_p'
    assert (args.communication_protocol in communication_protocol_list), "communication_protocol not available"

    args.Q = communication_protocol_dict['iteration_per_aggregation'] if (
                'iteration_per_aggregation' in communication_protocol_dict) else 1
    assert (args.Q % 1 == 0 and args.Q > 0), "iteration_per_aggregation should be positive integers"

    args.quant_level = communication_protocol_dict['quant_level'] if (
                'quant_level' in communication_protocol_dict) else 0
    args.vecdim = communication_protocol_dict['vecdim'] if ('vecdim' in communication_protocol_dict) else 1
    args.num_update_per_batch = communication_protocol_dict['num_update_per_batch'] if (
                'num_update_per_batch' in communication_protocol_dict) else 5
    args.num_batch_per_workset = communication_protocol_dict['num_batch_per_workset'] if (
                'num_batch_per_workset' in communication_protocol_dict) else 5
    args.smi_thresh = communication_protocol_dict['smi_thresh'] if (
                'smi_thresh' in communication_protocol_dict) else 0.5
    if args.quant_level > 0:
        args.ratio = math.log(args.quant_level, 2) / 32
    args.ratio = communication_protocol_dict['ratio'] if ('ratio' in communication_protocol_dict) else 0.5
    # print('Topk Ratio:',args.ratio)
    if args.communication_protocol == 'FedSGD':
        args.Q = 1
    print('communication_protocol:', args.communication_protocol)

    args.attacker_id = []
    # # args.early_stop, if use early stop
    # args.main_early_stop = config_dict['main_early_stop'] if ('main_early_stop' in config_dict) else 0
    # args.main_early_stop_param = config_dict['main_early_stop_param'] if ('main_early_stop_param' in config_dict) else 0.0001
    # # args.num_exp number of repeat experiments for main task
    # args.num_exp = config_dict['num_exp'] if ('num_exp' in config_dict) else 10

    # args.dataset_split  configuration for dataste
    args.dataset_split = config_dict['dataset'] if ('dataset' in config_dict) else None
    if args.dataset_split == None:
        args.num_classes = 10
        args.use_prompt = 0
        args.n_shot = 0
    args.num_classes = args.dataset_split['num_classes'] if ('num_classes' in args.dataset_split) else 10
    args.use_prompt = args.dataset_split['use_prompt'] if ('use_prompt' in args.dataset_split) else 0
    args.n_shot = args.dataset_split['n_shot'] if ('n_shot' in args.dataset_split) else 0

    ############## for LLM ###############
    args.pipeline = config_dict['pipeline'] if('pipeline' in config_dict) else None# pretrained finetune
    args.finetune_configs = config_dict['finetune_configs'] if('finetune_configs' in config_dict) else None
    args.finetune_name = args.finetune_configs['name'] if args.finetune_configs!=None else 'Vanilla'
    args.finetune_detail_configs = args.finetune_configs['configs'] if (args.finetune_configs!=None and 'configs' in args.finetune_configs)  else None

    args.model_architect = config_dict['model_architect'] if('model_architect' in config_dict) else 'CLM'
    print('load model_architect:',args.model_architect)

    # Tokenizer
    args.tokenizer = None  # for LLM if needed
    args.tokenizer_dict = config_dict['tokenizer'] if ('tokenizer' in config_dict) else None
    if args.tokenizer_dict != None:

        args.padding = args.tokenizer_dict['padding'] if ('padding' in args.tokenizer_dict) else 0
        # "longest"  "max_length"  "do_not_pad"
        args.padding_type = args.tokenizer_dict['padding_type'] if (
                    'padding_type' in args.tokenizer_dict) else 'outside'
        # "inside" "outside"
        args.pad_token = args.tokenizer_dict['pad_token'] if ('pad_token' in args.tokenizer_dict) else 'default'
        # default
        args.truncation = args.tokenizer_dict['truncation'] if (
                    'truncation' in args.tokenizer_dict) else "do_not_truncate"
        # "only first"  "only_second"  "longest_first"  "do_not_truncate"
        if args.truncation == "True":
            args.truncation = True
        elif args.truncation == "False":
            args.truncation = False

        args.max_length = args.tokenizer_dict['max_length'] if ('max_length' in args.tokenizer_dict) else None
        args.padding_side = args.tokenizer_dict['padding_side'] if ('padding_side' in args.tokenizer_dict) else "left"
        args.add_special_tokens = args.tokenizer_dict['add_special_tokens'] if (
                    'add_special_tokens' in args.tokenizer_dict) else 0
        if args.add_special_tokens == 0:
            args.add_special_tokens = False
        else:
            args.add_special_tokens = True
    else:
        args.padding = 'do_not_pad'

    if args.padding == 'do_not_pad':
        args.pad_info = f'donotpad'

    elif args.padding == 'longest':
        args.pad_info = f'longest-{str(args.pad_token)}-{str(args.truncation)}-{str(args.padding_side)}-{str(args.max_length)}-{str(args.add_special_tokens)}-{str(args.padding_type)}'
    else:
        args.pad_info = f'maxlength-{str(args.pad_token)}-{str(args.truncation)}-{str(args.padding_side)}-{str(args.max_length)}-{str(args.add_special_tokens)}-{str(args.padding_type)}'
    ############## for LLM ###############

    # args.model_list, specify the types of models
    if 'model_list' in config_dict:
        config_model_dict = config_dict['model_list']
        # print('config_model_dict:',(len(config_model_dict)-2))
        # assert ((len(config_model_dict)-3)==args.k), 'please alter party number k, model number should be equal to party number'

        ############## for LLM ###############
        args.max_sequence = config_model_dict['0']['max_sequence'] if ('max_sequence' in config_model_dict['0']) else -1

        model_dict = {}
        default_dict_element = {'type': 'MLP2', 'path': 'random_14*28_10', 'input_dim': 392, 'output_dim': 10}

        # LLM Task Description
        args.task_dict = config_model_dict['task'] if('task' in config_model_dict) else None
        if args.task_dict != None:
            args.task_type = args.task_dict['task_type'] if('task_type' in args.task_dict) else "SequenceClassification"
            args.generation_config_dict = args.task_dict['generation_config_dict'] if('generation_config_dict' in args.task_dict) else {}
            args.metric_type = args.task_dict['metric_type'] if('metric_type' in args.task_dict) else "n_best"
            args.doc_stride = args.task_dict['doc_stride'] if('doc_stride' in args.task_dict) else -1
            args.max_query_length = args.task_dict['max_query_length'] if('max_query_length' in args.task_dict) else -1
            args.max_seq_length = args.task_dict['max_seq_length'] if('max_seq_length' in args.task_dict) else -1
            args.max_answer_length = args.task_dict['max_answer_length'] if('max_answer_length' in args.task_dict) else -1
            args.n_best_size = args.task_dict['n_best_size'] if('n_best_size' in args.task_dict) else 20
            args.max_new_tokens = args.task_dict['max_new_tokens'] if('max_new_tokens' in args.task_dict) else 1
        else:
            args.task_type = None
            args.generation_config_dict = None
            args.metric_type = None
            args.doc_stride = None
            args.max_query_length = None
            args.max_seq_length = None
            args.max_answer_length = None
            args.n_best_size = None
            args.max_new_tokens = None
        
        args.encoder_trainable = []
        args.embedding_trainable = []
        args.head_layer_trainable = []
        args.encoder_trainable_ids_list = []

        for ik in range(args.k):
            if str(ik) in config_model_dict:
                if 'type' in config_model_dict[str(ik)]:
                    args.model_type = config_model_dict[str(ik)]['model_type'] if 'model_type' in config_model_dict[
                        str(ik)] else None  # Overall Model Type LLM type/None for non-llm scenario

                    encoder_trainable_ids = config_model_dict[str(ik)]['encoder_trainable_ids'] if 'encoder_trainable_ids' in config_model_dict[str(ik)] else []
                    args.encoder_trainable_ids_list.append(encoder_trainable_ids)

                    if ik == args.k - 1:
                        args.embedding_trainable.append(False)  # no embedding layer for active parties
                    else:
                        embedding_trainable = int(
                            config_model_dict[str(ik)]['embedding_trainable']) if 'embedding_trainable' in \
                                                                                  config_model_dict[
                                                                                      str(ik)] else 0  # Overall Model Type
                        if embedding_trainable == 1:
                            embedding_trainable = True
                        else:
                            embedding_trainable = False
                        args.embedding_trainable.append(embedding_trainable)

                    encoder_trainable = int(config_model_dict[str(ik)]['encoder_trainable']) if 'encoder_trainable' in \
                                                                                                config_model_dict[
                                                                                                    str(ik)] else 0  # Overall Model Type
                    if encoder_trainable == 1:
                        encoder_trainable = True
                    else:
                        encoder_trainable = False
                    args.encoder_trainable.append(encoder_trainable)

                    if ik != args.k - 1:
                        args.head_layer_trainable.append(False)  # no head layer for passive parties
                    else:
                        head_layer_trainable = int(config_model_dict[str(ik)]['head_layer_trainable']) if 'head_layer_trainable' in config_model_dict[str(ik)] else 0
                        if head_layer_trainable == 1:
                            head_layer_trainable = True
                        else:
                            head_layer_trainable = False
                        args.head_layer_trainable.append(head_layer_trainable)

                    if 'path' in config_model_dict[str(ik)] or (('input_dim' in config_model_dict[str(ik)]) and (
                            'output_dim' in config_model_dict[str(ik)])):
                        model_dict[str(ik)] = config_model_dict[str(ik)]
                        args.model_path = config_model_dict[str(ik)]['path']
                        args.pretrained = config_model_dict[str(ik)]['pretrained'] if 'pretrained' in config_model_dict[str(ik)] else 1
                    else:
                        model_type_name = config_model_dict[str(ik)]['type']  # specific model type
                        temp = {'type': model_type_name, 'path': '../models/' + model_type_name + '/random'}
                        model_dict[str(ik)] = temp
                        args.model_path = ""
                        args.pretrained = 1
                else:
                    model_dict[str(ik)] = default_dict_element
            else:
                model_dict[str(ik)] = default_dict_element

        print('args.encoder_trainable:', args.encoder_trainable)
        print('args.embedding_trainable:', args.embedding_trainable)
        print('args.head_layer_trainable:', args.head_layer_trainable)

        args.model_list = model_dict
        args.local_encoders_num = config_model_dict[
            'local_encoders_num'] if 'local_encoders_num' in config_model_dict else 1
        args.apply_trainable_layer = config_model_dict['apply_trainable_layer'] if (
                    'apply_trainable_layer' in config_model_dict) else 0
        args.global_model = config_model_dict['global_model'] if (
                    'global_model' in config_model_dict) else 'ClassificationModelHostHead'
        print('args.local_encoders_num:', args.local_encoders_num)

    else:
        default_model_dict = {}
        default_dict_element = {'type': 'MLP2', 'path': '../models/MLP2/random'}
        for ik in range(args.k):
            default_model_dict[str(ik)] = default_dict_element
        args.model_list = default_model_dict
        args.apply_trainable_layer = 0
        args.global_model = 'ClassificationModelHostHead'

    # Check: Centralized Training
    if args.k == 1:
        print('k=1, Launch Centralized Training, All Attack&Defense dismissed, Q set to 1')
        args.apply_attack = False  # bli/ns/ds attack
        args.apply_backdoor = False  # replacement backdoor attack
        args.apply_nl = False  # noisy label attack
        args.apply_ns = False  # noisy sample attack
        args.apply_mf = False  # missing feature attack
        args.apply_defense = False
        args.apply_mid = False
        args.apply_cae = False
        args.apply_dcae = False
        args.apply_dp = False
        args.Q = 1
        # return args

    # if defense appears
    args.apply_defense = False
    args.apply_dp = False
    args.apply_mid = False  # mid defense
    args.apply_cae = False  # cae defense
    args.apply_dcae = False  # dcae defense
    args.apply_adversarial = False  # adversarial
    args.bin_size = [None for _ in range(args.k)]  # for discrete bins
    args.gradients_res_a = [None for _ in range(args.k)]  # for gradient sparsification
    args.apply_dcor = False  # distance corrilation
    if 'defense' in config_dict:
        print(config_dict['defense'].keys())
        if 'name' in config_dict['defense']:
            args.apply_defense = True
            args.defense_name = config_dict['defense']['name']
            args.defense_configs = config_dict['defense']['parameters'] if (
                        'parameters' in config_dict['defense']) else None
            assert 'party' in config_dict['defense']['parameters'], '[Error] Defense party not specified'
            print(f"in load configs, defense_configs is type {type(args.defense_configs)}")
            print(f"in load configs, defense_configs is type {type(dict(args.defense_configs))}")
            if 'mid' in args.defense_name.casefold():
                args.apply_mid = True
            elif 'cae' in args.defense_name.casefold():
                args.apply_cae = True
                if 'dcae' in args.defense_name.casefold():
                    args.apply_dcae = True
            elif 'adversarial' in args.defense_name.casefold():
                args.apply_adversarial = True
            elif 'distancecorrelation' in args.defense_name.casefold():
                args.apply_dcor = True
            elif ('gaussian' in args.defense_name.casefold()) or ('laplace' in args.defense_name.casefold()):
                args.apply_dp = True
        else:
            assert 'name' in config_dict['defense'], "missing defense name"
    else:
        args.defense_name = 'None'
        args.defense_configs = None
        print('===== No Defense ======')
    # get Info: args.defense_param  args.defense_param_name
    if args.apply_defense == True:
        if args.defense_name in ["CAE", "DCAE", "DistanceCorrelation", "AdversarialTraining"]:
            args.defense_param = args.defense_configs['lambda']
            args.defense_param_name = 'lambda'
        elif args.defense_name in ["MID"]:
            mid_model_name = str(args.defense_configs['mid_model_name']) if 'mid_model_name' in args.defense_configs else 'MID'
            mid_position = str(args.defense_configs['mid_position']) if 'mid_position' in args.defense_configs else 'out'
            args.defense_param = mid_model_name + '_' + mid_position+ '_' + str(args.defense_configs['lambda'])
            args.defense_param_name = 'lambda'
        elif args.defense_name == "GaussianDP" or args.defense_name == "LaplaceDP":
            if 'dp_strength' in args.defense_configs:
                args.defense_param = args.defense_configs['dp_strength']
                args.defense_param_name = 'dp_strength'
            else:
                args.defense_param = args.defense_configs['epsilon']
                args.defense_param_name = 'epsilon'
        elif args.defense_name == "GradientSparsification":
            args.defense_param = args.defense_configs['gradient_sparse_rate']
            args.defense_param_name = 'gradient_sparse_rate'
        elif args.defense_name == "GradPerturb":
            args.defense_param = args.defense_configs['perturb_epsilon']
            args.defense_param_name = 'perturb_epsilon'
        else:
            args.defense_param = 'None'
            args.defense_param_name = 'No_Defense'
    else:
        args.defense_param = 'None'
        args.defense_param_name = 'No_Defense'

    # Detail Specifications about defense_name
    if args.defense_name == "MID":
        if args.defense_configs['party'] == [0]:
            args.defense_name = "MID_Passive"
        else:
            args.defense_name = "MID_Active"

    # if there's attack   Mark attack type
    args.attack_num = 0
    args.targeted_backdoor_list = []
    args.targeted_backdoor_index = []
    args.untargeted_backdoor_list = []
    args.untargeted_backdoor_index = []
    args.label_inference_list = []
    args.label_inference_index = []
    args.attribute_inference_list = []
    args.attribute_inference_index = []
    args.feature_inference_list = []
    args.feature_inference_index = []
    args.inversion_list = []
    args.inversion_index = []
    args.apply_attack = False
    if 'attack_list' in config_dict:
        if len(config_dict['attack_list']) > 0:
            attack_config_dict = config_dict['attack_list']
            args.attack_num = len(attack_config_dict)
            args.apply_attack = True
            for ik in range(args.attack_num):
                if 'name' in attack_config_dict[str(ik)]:
                    _name = attack_config_dict[str(ik)]['name']
                    if _name in TARGETED_BACKDOOR:
                        args.targeted_backdoor_list.append(_name)
                        args.targeted_backdoor_index.append(ik)

                    elif _name in UNTARGETED_BACKDOOR:
                        args.untargeted_backdoor_list.append(_name)
                        args.untargeted_backdoor_index.append(ik)

                    elif _name in LABEL_INFERENCE:
                        args.label_inference_list.append(_name)
                        args.label_inference_index.append(ik)

                    elif _name in ATTRIBUTE_INFERENCE:
                        args.attribute_inference_list.append(_name)
                        args.attribute_inference_index.append(ik)

                    elif _name in FEATURE_INFERENCE:
                        args.feature_inference_list.append(_name)
                        args.feature_inference_index.append(ik)

                    # LLM attacks
                    elif _name in INVERSION:
                        args.inversion_list.append(_name)
                        args.inversion_index.append(ik)
                else:
                    assert 'name' in attack_config_dict[str(ik)], 'missing attack name'
        else:
            assert len(config_dict['attack_list']) > 0, 'empty attack_list'
    else:
        print('===== No Attack ======')

    return args


def load_attack_configs(config_file_name, args, index):
    '''
    load attack[index] in attack_list
    '''
    config_file_path = './configs/' + config_file_name + '.json'
    config_file = open(config_file_path, "r")
    config_dict = json.load(config_file)

    args.attack_type = None
    args.apply_backdoor = False  # replacement backdoor attack
    args.apply_nl = False  # noisy label attack
    args.apply_ns = False  # noisy sample attack
    args.apply_mf = False  # missing feature attack

    # No Attack
    if index == -1:
        print('No Attack==============================')
        args.attack_name = 'No_Attack'
        args.attack_param_name = 'None'
        args.attack_param = None
        return args

    # init args about attacks
    assert args.apply_attack == True
    # choose attack[index]
    attack_config_dict = config_dict['attack_list'][str(index)]

    args.attaker_id = attack_config_dict['party'] if ('party' in attack_config_dict) else []

    if 'name' in attack_config_dict:
        args.attack_name = attack_config_dict['name']
        args.attack_configs = attack_config_dict['parameters'] if ('parameters' in attack_config_dict) else None

        if args.attack_name in TARGETED_BACKDOOR:
            args.attack_type = 'targeted_backdoor'
            if 'backdoor' in args.attack_name.casefold():
                args.apply_backdoor = True

        elif args.attack_name in UNTARGETED_BACKDOOR:
            args.attack_type = 'untargeted_backdoor'
            if 'noisylabel' in args.attack_name.casefold():
                args.apply_nl = True
            if 'noisysample' in args.attack_name.casefold():
                args.apply_ns = True
            if 'missingfeature' in args.attack_name.casefold():
                args.apply_mf = True

        elif args.attack_name in LABEL_INFERENCE:
            args.attack_type = 'label_inference'

        elif args.attack_name in ATTRIBUTE_INFERENCE:
            args.attack_type = 'attribute_inference'

        elif args.attack_name in FEATURE_INFERENCE:
            args.attack_type = 'feature_inference'

        elif args.attack_name in INVERSION:
            args.attack_type = 'inversion'

        else:
            assert 0, 'attack type not supported'

        if args.attack_name == 'NoisyLabel':
            args.attack_param_name = 'noise_type'
            args.attack_param = str(attack_config_dict['parameters']['noise_type']) + '_' + str(
                attack_config_dict['parameters']['noise_rate'])
        elif args.attack_name == 'MissingFeature':
            args.attack_param_name = 'missing_rate'
            args.attack_param = str(attack_config_dict['parameters']['missing_rate'])
        elif args.attack_name == 'BatchLabelReconstruction':
            args.attack_param_name = 'attack_lr'
            args.attack_param = str(attack_config_dict['parameters']['lr'])
        elif args.attack_name == 'NoisySample':
            args.attack_param_name = 'noise_lambda'
            args.attack_param = str(attack_config_dict['parameters']['noise_lambda'])
        else:
            args.attack_param_name = 'None'
            args.attack_param = None

    else:
        assert 'name' in attack_config_dict, "missing attack name"

    # Check: Centralized Training
    if args.k == 1:
        print('k=1, Launch Centralized Training, All Attack&Defense dismissed, Q set to 1')
        args.apply_attack = False  # bli/ns/ds attack
        args.apply_backdoor = False  # replacement backdoor attack
        args.apply_nl = False  # noisy label attack
        args.apply_ns = False  # noisy sample attack
        args.apply_mf = False  # missing feature attack
        args.apply_defense = False
        args.apply_mid = False
        args.apply_cae = False
        args.apply_dcae = False
        args.apply_dp = False
        args.Q = 1

    return args


def init_attack_defense(args):
    args.apply_attack = False
    args.apply_backdoor = False  # replacement backdoor attack
    args.apply_nl = False  # noisy label attack
    args.apply_ns = False  # noisy sample attack
    args.apply_mf = False  # missing feature attack
    args.apply_defense = False
    args.apply_mid = False
    args.apply_cae = False
    args.apply_dcae = False
    args.apply_dp = False
    return args


if __name__ == '__main__':
    pass
