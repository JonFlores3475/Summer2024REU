import numpy as np
from Methods.utils.meta_methods import FederatedMethod
from Aggregations import Aggregation_NAMES
from Attack.backdoor.utils import backdoor_attack
from Attack.byzantine.utils import attack_dataset
from Attack.Poisoning_Attack.utils import inverted_gradient
from Datasets.federated_dataset.single_domain import single_domain_dataset_name, get_single_domain_dataset
from Methods import Fed_Methods_NAMES, get_fed_method
from utils.conf import set_random_seed, config_path
from Datasets.federated_dataset.multi_domain import multi_domain_dataset_name, get_multi_domain_dataset
from Backbones import get_private_backbones
from utils.cfg import CFG as cfg, simplify_cfg,show_cfg
from utils.utils import ini_client_domain
from argparse import ArgumentParser
from utils.training import train
import setproctitle
import argparse
import datetime
import socket
import uuid
import copy
import os
import sys

# General Notes:
# When the attack is a backdoor attack, it creates a partial_cfg variable, updating its information, and including it
# into the function call. This initially made me through that the "CfgNode" is the "local model". However, it is also
# used in the train method, which is only called once rather than multiple times. So CfgNode isn't the local model? Idk
# overall I'm still a little confused about it.

# Parses the arguments that are passed in when the program is started. It takes different flags and seeing if they were
# in the line, setting a value if so.
#
# return - returns the parsed arguments
def parse_args():
    # Creates the parser to start parsing the arguments
    parser = ArgumentParser(description='Federated Learning', allow_abbrev=False)
    # Adds flag for device ID, having a default value of 7
    parser.add_argument('--device_id', type=int, default=7, help='The Device Id for Experiment')
    '''
    Task: OOD label_skew domain_skew
    '''
    # Adds flag for the type of task, with the default being domain_skew
    parser.add_argument('--task', type=str, default='domain_skew')
    '''
    label_skew:   fl_cifar10 fl_cifar100 fl_mnist fl_usps fl_fashionmnist fl_tinyimagenet
    domain_skew: Digits,OfficeCaltech, PACS PACScomb OfficeHome Office31 VLCS
    '''
    # Adds flag for the type of dataset, with Digits being the default (make sure the dataset matches the type of task)
    parser.add_argument('--dataset', type=str, default='Digits',
                        help='Which scenario to perform experiments on.')
    '''
    Attack: byzantine backdoor Poisoning_Attack None
    '''
    # Adds flag for the attack type, with the default being none
    parser.add_argument('--attack_type', type=str, default='None')

    '''
    Extra Attack Flags
    '''
    # Adds general attack flags to be used for either byzantine or backdoor attacks
    # bad_client_rate and noise_data_rate, each with its own default value
    parser.add_argument('--bad_client_rate', type=float, default=0.2, help='The ratio of bad clients')
    parser.add_argument('--noise_data_rate', type=float, default=0.5, help='Rate of noise')

    '''
    Extra Byzantine Attack Flags
    '''
    # Adds flags for byzantine_evils, dev_type, and lamda, each with its own default value
    parser.add_argument('--byzantine_evils', type=str, default='PairFlip', 
                        help='Which type of byzantine attack: PairFlip, SymFlip, RandomNoise, lie_attack, min_max, or min_sum')
    parser.add_argument('--dev_type', type=str, default='std', help='Parameter for min_max and min_sum')
    parser.add_argument('--lamda', type=float, default=10.0, help='Parameter for min_max and min_sum')
    
    '''
    Extra Backdoor Attack Flags
    '''
    # Adds flags for backdoor_evils, backdoor_label, and semantic_backdoor_label, each having their own default value.
    parser.add_argument('--backdoor_evils', type=str, default='base_backdoor', 
                        help='Which type of backdoor attack: base_backdoor, semantic_backdoor, sneaky_backdoor, gaus_images, shrink_stretch, sneaky_random, sneaky_random2, sneaky_random3, sneaky_random4, sneaky_random5, or atropos (NOTE: ONLY FOR USE IN POISONING ATTACK)')
    parser.add_argument('--backdoor_label', type=int, default=2, help='Which label to change (int)')
    parser.add_argument('--semantic_backdoor_label', type=int, default=3, help='Which label to change to (int)')

    # Adds a flag for poisoning_evils, with its default value
    parser.add_argument('--poisoning_evils', type=str, default='inverted_gradient', 
                        help='Which type of Poisoning attack: inverted_gradient')
    
    '''
    Federated Method: FedRC FedAVG FedR FedProx FedDyn FedOpt FedProc FedR FedProxRC  FedProxCos FedNTD  DelphiflMedian DelphiflZeroTrust DelphiflZeroTrustV2
    '''
    # Adds flag for the type of method the federated learning model is using, with qffeAVG being the default
    parser.add_argument('--method', type=str, default='qffeAVG',
                        help='Federated Method name.', choices=Fed_Methods_NAMES)
    # Adds flag for the random_domain_select, setting to to False as a default
    parser.add_argument('--rand_domain_select', type=bool, default=False, help='The Local Domain Selection')
    # Adds flag for the type of structure the federated learning model is
    parser.add_argument('--structure', type=str, default='homogeneity')  # 'homogeneity' heterogeneity

    '''
    Aggregations Strategy Hyper-Parameter
    '''
    # Adds flag for the averaging, with Weight being the default
    parser.add_argument('--averaging', type=str, default='Weight', choices=Aggregation_NAMES, help='The Option for averaging strategy')
    # Weight Equal

    # Adds flag for the seed, being 0 as a default
    parser.add_argument('--seed', type=int, default=0, help='The random seed.')

    # Adds flag for the csv_log to print logs to a csv file, being set as False by default
    parser.add_argument('--csv_log', action='store_true', default=False, help='Enable csv logging')
    # Adds flag for the csv_name to name the file where the logs will be logged at, being set to None by default.
    parser.add_argument('--csv_name', type=str, default=None, help='Predefine the csv name')
    # Adds flag for save_checkpoint, being set to False by default.
    parser.add_argument('--save_checkpoint', action='store_true', default=False)
    # Adds flag for opts, being set to None by default.
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

    # Uses the parsers parse_args() function to parse the arguments.
    args = parser.parse_args()
    # Returns the parsed arguments
    return args

# The main method of the program. This runs the simulation: the training phase, testing phase, any attacks that may be being
# utilized, any changes in the values, etc.
#
# args - the arguments being passed in
def main(args=None):
    client_type = np.array([True])
    # If the args are none, then parse the arguments
    if args is None:
        args = parse_args()
    
    # TODO: Check that none of the arguments conflict or would be invalid

    # Sets a bunch of the initial values of the arguments (timestamp, host, path, etc.)
    args.conf_jobnum = str(uuid.uuid4())
    args.conf_timestamp = str(datetime.datetime.now())
    args.conf_host = socket.gethostname()

    cfg_dataset_path = os.path.join(config_path(), args.task, args.dataset, 'Default.yaml')
    cfg.merge_from_file(cfg_dataset_path)

    cfg_method_path = os.path.join(config_path(), args.dataset, args.method + '.yaml')
    # Checks to see if the path exists, merging it from a file if it does
    if os.path.exists(cfg_method_path):
        cfg.merge_from_file(cfg_method_path)

    cfg.merge_from_list(args.opts)

    # Creates the particial CfgNode using the arguments as settings
    particial_cfg = simplify_cfg(args, cfg)

    # If there is an attack being carried out, set the bad_client_rate and noise_data_rate
    if args.attack_type != 'None':
        particial_cfg.attack.bad_client_rate = args.bad_client_rate
        particial_cfg.attack.noise_data_rate = args.noise_data_rate
        
        if args.attack_type == 'byzantine':
            '''
            Updating Additional Attack Flags
            '''
            # Sets the particial_cfg's variables to the arguments' variables dev_type, and lamda,
            particial_cfg.attack.byzantine.evils = args.byzantine_evils
            particial_cfg.attack.byzantine.dev_type = args.dev_type
            particial_cfg.attack.byzantine.dev_type = args.lamda
        
        elif args.attack_type == 'backdoor':
            '''
            Updating Additional Attack Flags
            '''
            # Sets the particial_cfg's variables to the arguments' variables
            particial_cfg.attack.backdoor.evils = args.backdoor_evils
            particial_cfg.attack.backdoor.backdoor_label = args.backdoor_label
            particial_cfg.attack.backdoor.semantic_backdoor_label = args.semantic_backdoor_label

    # Prints the cfg to make sure all the information is correct
    show_cfg(args,particial_cfg,args.method)
    # If the seed is not none, it sets the random seed as the seed brought in through the arguments
    if args.seed is not None:
        set_random_seed(args.seed)

    '''
    Loading the dataset
    '''
    if args.dataset in multi_domain_dataset_name:
        private_dataset = get_multi_domain_dataset(args, particial_cfg)
    elif args.dataset in single_domain_dataset_name:
        private_dataset = get_single_domain_dataset(args, particial_cfg)

    if args.task == 'OOD':
        '''
        Define clients domain
        '''
        in_domain_list = copy.deepcopy(private_dataset.domain_list)
        # If the out_domain at cfg[args.task] is not 'NONE', then it removes it from the in_domain_list and sets the 
        # private dataset's in_domain_list to the newly made in_domain_list
        if cfg[args.task].out_domain != "NONE":
            in_domain_list.remove(cfg[args.task].out_domain)
            private_dataset.in_domain_list = in_domain_list

        private_dataset.in_domain_list = in_domain_list  # 参与者能够从哪几个Domain中获取数据

        # Gets the temporary client domain list
        temp_client_domain_list = ini_client_domain(args.rand_domain_select, private_dataset.domain_list, particial_cfg.DATASET.parti_num)


        client_domain_list = []
        # For i in the range of temp_client_domain_list's length . . . 
        for i in range(len(temp_client_domain_list)):
            # If at the i-th index of temp_client_domain_list does NOT equal the out_domain at the cfg[args.task], it appends
            # the object at the i-th index of temp_client_domain_list to the client_domain_list
            if temp_client_domain_list[i] != cfg[args.task].out_domain:
                client_domain_list.append(temp_client_domain_list[i])

        # Sets the particial_cfg's dataset's participant number to the length of the client domain list
        particial_cfg.DATASET.parti_num = len(client_domain_list)

        # Sets the private dataset's client_domain_list to the already found client_domain_list
        private_dataset.client_domain_list = client_domain_list
        # Appends the cfg[args.task]'s out_domain to the client_domain_list
        client_domain_list.append(cfg[args.task].out_domain)

        # Gets the data_loaders with respect to the client_domain_list
        private_dataset.get_data_loaders(client_domain_list)
        # Pops the first set of train_loaders off of the domain_list, setting that to the out_train_loader
        private_dataset.out_train_loader = private_dataset.train_loaders.pop()
        # Pops the fist object from the client_domain_list
        client_domain_list.pop()
    # Else, if the arguments' task is 'label_skew'
    elif args.task == 'label_skew':
        # gets the data loaders and sets the client_domain_list to None
        private_dataset.get_data_loaders()
        client_domain_list = None
    # Else, if the arguments' task is 'domain_skew'
    elif args.task == 'domain_skew':
        # Gets the client_domain_list and gets the data_loaders off of the domain_list
        client_domain_list = ini_client_domain(args.rand_domain_select, private_dataset.domain_list, particial_cfg.DATASET.parti_num)
        private_dataset.get_data_loaders(client_domain_list)

    # If the attack_type is 'byzantine'
    if args.attack_type == 'byzantine':
        # Checks to see if the dataset is a multi_domain_dataset, setting it as so
        if args.dataset in multi_domain_dataset_name:
            particial_cfg.attack.dataset_type = 'multi_domain'
        # Else, if it is a single_domain_dataset, it sets it as so
        elif args.dataset in single_domain_dataset_name:
            particial_cfg.attack.dataset_type = 'single_domain'
        
        # Gets the bad scale, casting it as an integer
        bad_scale = int(particial_cfg.DATASET.parti_num * particial_cfg['attack'].bad_client_rate)
        # Gets the good scale based off of the bad scale
        good_scale = particial_cfg.DATASET.parti_num - bad_scale
        # Gets the client type
        client_type = np.repeat(True, good_scale).tolist() + (np.repeat(False, bad_scale)).tolist()
    
        # Uses the attack_dataset
        attack_dataset(args, particial_cfg, private_dataset, client_type)

    # Else, if the attack_type is 'backdoor'
    elif args.attack_type == 'backdoor':
        # Gets the bad scale
        bad_scale = int(particial_cfg.DATASET.parti_num * particial_cfg['attack'].bad_client_rate)
        # Gets the good scale based off of the bas scale
        good_scale = particial_cfg.DATASET.parti_num - bad_scale
        # Gets the client type
        client_type = np.repeat(True, good_scale).tolist() + (np.repeat(False, bad_scale)).tolist()

        # Atropos can only be used when the attack_type is 'Poisoning_Attack' and the poisoning_evils are 'inverted_gradient'
        # If this is not the case, the attack will not function properly
        if args.backdoor_evils == 'atropos': 
            print("ERROR: Atropos must be used with Poisoning_Attack as the attack_type and inverted_gradient as the poisoning_evils")
            sys.exit()

        # Does a backdoor attack during the training phase
        backdoor_attack(args, particial_cfg, client_type, private_dataset, is_train=True)
        # Does another backdoor attack not during the training phase
        backdoor_attack(args, particial_cfg, client_type, private_dataset, is_train=False)
    
    # Else, if the attack_type is 'Poisoning_Attack'
    elif args.attack_type == "Poisoning_Attack":
        particial_cfg.attack.Poisoning_Attack.evils = args.poisoning_evils
        # Gets the bad scale
        bad_scale = int(particial_cfg.DATASET.parti_num * particial_cfg['attack'].bad_client_rate)
        # Gets the good scale based off of the bad scale
        good_scale = particial_cfg.DATASET.parti_num - bad_scale
        # Gets the client type
        client_type = np.repeat(True, good_scale).tolist() + (np.repeat(False, bad_scale)).tolist()
        print(client_type)

        if cfg[args.method].local_method != 'FedProxLocal':
            print("ERROR: all poisoning attacks must be used with FexProxLocal as the local method")
            sys.exit()
        
        if args.backdoor_evils == 'atropos':
            # Does an attack during the training phase
            backdoor_attack(args, particial_cfg, client_type, private_dataset, is_train=True)
            # Does another attack not during the training phase
            backdoor_attack(args, particial_cfg, client_type, private_dataset, is_train=False)


    '''
    Loading the Private Backbone
    '''
    priv_backbones = get_private_backbones(particial_cfg)

    '''
    Loading the Federated Optimizer
    '''

    fed_method = get_fed_method(priv_backbones, client_domain_list, args, particial_cfg)
    assert args.structure in fed_method.COMPATIBILITY

    # Checks to see if the task is OOD
    if args.task == 'OOD':
        # Sets the fed_method's train_eval_loaders equal to the private dataset's train_eval_loaders
        fed_method.train_eval_loaders = private_dataset.train_eval_loaders
        # Does the same thing with the test_loaders as line 256
        fed_method.test_loaders = private_dataset.test_loader
    # Checks to see if the attack type was 'byzantine
    if args.attack_type == 'byzantine':
        # Sets the fed_method's client_type to the recorded client_type 
        fed_method.client_type = client_type

    # If the csv_name is None (so no csv_file)
    if args.csv_name is None:
        # Sets the proctitle of everything
        setproctitle.setproctitle('{}_{}_{}'.format(args.method, args.task,args.dataset))
    # If the csv_name is NOT none
    else:
        # Sets the proctitle of everything, including the csv_name
        setproctitle.setproctitle('{}_{}_{}_{}'.format(args.method, args.task,args.dataset, args.csv_name))
    # It then trains the model based off of the fed_method, private_dataset, the arguments, the current particial_cfg,
    # and the client_domain_list
    train(fed_method, private_dataset, args, particial_cfg, client_domain_list, client_type)

# If the name of the file that was executed is '__main__', then it calls the main() method
if __name__ == '__main__':
    main()
