import copy
import os
import csv
from utils.conf import log_path
from utils.utils import create_if_not_exists

import yaml
from yacs.config import CfgNode as CN

# Arguments that are not used in this code and are removed for simplicity's sake later
except_args = ['csv_log', 'csv_name', 'device_id', 'seed', 'tensorboard', 'conf_jobnum', 'conf_timestamp', 'conf_host', 'opts']


# Class used to write the log for the simulation
#
# args - the arguments passed in through main
# cfg - the CFG node being passed in
class CsvWriter:
    def __init__(self, args, cfg):
        self.args = args
        self.cfg = cfg
        self.model_path = self.model_folder_path()
        self.para_path = self.write_para()
        print(self.para_path)

    # Making an OS path to get to the model
    def model_folder_path(self):
        if self.args.task == 'OOD':
            model_path = os.path.join(log_path(), self.args.task, self.args.attack_type, self.args.dataset, self.cfg.OOD.out_domain, self.args.averaging,
                                      self.args.method)
        else:
            if self.args.attack_type == 'None':
                model_path = os.path.join(log_path(), self.args.task, self.args.attack_type, self.args.dataset, self.args.averaging, self.args.method)
            else:
                model_path = os.path.join(log_path(), self.args.task, self.cfg.attack[self.args.attack_type].evils,
                                          self.args.dataset, self.args.averaging, self.args.method)
        create_if_not_exists(model_path)
        return model_path

    # Used in training.py to write the weights in the different epochs
    #
    # weight_dict - dictionary of the weights, gets written to the write file
    # epoch_index = used to know what and how to write
    # client_domain_list - list, gets written to the write file
    def write_weight(self, weight_dict, epoch_index, client_domain_list):
        weight_path = os.path.join(self.para_path, 'weight.csv')
        if epoch_index != 0:
            write_type = 'a'
        else:
            write_type = 'w'

        with open(weight_path, write_type) as result_file:
            result_file.write(str(epoch_index) + ':' + '\n')
            for i in range(len(client_domain_list)):
                result_file.write(client_domain_list[i] + ',')
            result_file.write('\n')
            for k in weight_dict:
                result_file.write(k + ':' + str(list(weight_dict[k])) + '\n')

    # Makes an account path, either ALL or MEAN
    def write_acc(self, acc, name, mode='ALL'):
        if mode == 'ALL':
            acc_path = os.path.join(self.para_path, name + '_all_acc.csv')
            self.write_all_acc(acc_path, acc)
        elif mode == 'MEAN':
            mean_acc_path = os.path.join(self.para_path, name + '_mean_acc.csv')
            self.write_mean_acc(mean_acc_path, acc)

    # Adds cfg to a dictionary and returns it
    def cfg_to_dict(self, cfg):
        d = {}
        for k, v in cfg.items():
            if isinstance(v, CN):
                d[k] = self.cfg_to_dict(v)
            else:
                d[k] = v
        return d

    # Adds a dictionary to the cfg and returns it
    def dict_to_cfg(self, d):
        cfg = CN()
        for k, v in d.items():
            if isinstance(v, dict):
                cfg[k] = self.dict_to_cfg(v)
            else:
                cfg[k] = v
        return cfg

    # Write the parameters of the model to the log
    def write_para(self) -> None:
        from yacs.config import CfgNode as CN

        args = copy.deepcopy((self.args))
        args = vars(args)
        cfg = copy.deepcopy(self.cfg)

        # Remove the excepted args from the list of arguments
        for cc in except_args:
            if cc in args: del args[cc]
        for key, value in args.items():
            args[key] = str(value)
        paragroup_dirs = os.listdir(self.model_path)
        n_para = len(paragroup_dirs)
        final_check = False

        # If a specific name is included in the args, use it in the path
        # Otherwise, make a seperate path not including a name
        # Return the resulting path
        if self.args.csv_name is not None:

            path = os.path.join(self.model_path, self.args.csv_name)
            create_if_not_exists(path)
            args_path = path + '/args.csv'
            cfg_path = path + '/cfg.yaml'
            columns = list(args.keys())
            with open(args_path, 'w') as tmp:
                writer = csv.DictWriter(tmp, fieldnames=columns)

                writer.writeheader()
                writer.writerow(args)
            with open(cfg_path, 'w') as f:
                f.write(yaml.dump(self.cfg_to_dict(cfg)))
        else:

            for para in paragroup_dirs:
                exist_para_args = True
                exist_para_cfg = True
                dict_from_csv = {}
                key_value_list = []
                para_path = os.path.join(self.model_path, para)
                args_path = para_path + '/args.csv'
                with open(args_path, mode='r') as inp:
                    reader = csv.reader(inp)
                    for rows in reader:
                        key_value_list.append(rows)
                for index, _ in enumerate(key_value_list[0]):
                    dict_from_csv[key_value_list[0][index]] = key_value_list[1][index]
                if args != dict_from_csv:
                    exist_para_args = False
                cfg_path = para_path + '/cfg.yaml'
                query_cfg = copy.deepcopy(cfg)
                query_cfg.merge_from_file(cfg_path)
                for name, value1 in cfg.items():
                    if isinstance(value1, CN):
                        if name not in query_cfg or self.cfg_to_dict(query_cfg[name]) != self.cfg_to_dict(value1):
                            exist_para_cfg = False
                if exist_para_args == True and exist_para_cfg == True:
                    final_check = True
                    break

            if not final_check:

                if self.args.csv_name is None:
                    path = os.path.join(self.model_path, 'para' + str(n_para + 1))
                    k = 1
                    while os.path.exists(path):
                        path = os.path.join(self.model_path, 'para' + str(n_para + k))
                        k = k + 1
                else:
                    path = os.path.join(self.model_path, self.args.csv_name)

                create_if_not_exists(path)
                columns = list(args.keys())
                write_headers = True
                args_path = path + '/args.csv'
                cfg_path = path + '/cfg.yaml'
                with open(args_path, 'a') as tmp:
                    writer = csv.DictWriter(tmp, fieldnames=columns)
                    if write_headers:
                        writer.writeheader()
                    writer.writerow(args)
                with open(cfg_path, 'w') as f:
                    f.write(yaml.dump(self.cfg_to_dict(cfg)))
            else:
                path = para_path
        return path

    # Write the mean results of the accounts
    #
    # mean_path - path to the mean value
    # acc_list - list of accounts
    def write_mean_acc(self, mean_path, acc_list):
        if os.path.exists(mean_path):
            with open(mean_path, 'a') as result_file:
                for i in range(len(acc_list)):
                    result = acc_list[i]
                    result_file.write(str(result))
                    if i != self.cfg.DATASET.communication_epoch - 1:
                        result_file.write(',')
                    else:
                        result_file.write('\n')
        else:
            with open(mean_path, 'w') as result_file:
                for epoch in range(self.cfg.DATASET.communication_epoch):
                    result_file.write('epoch_' + str(epoch))
                    if epoch != self.cfg.DATASET.communication_epoch - 1:
                        result_file.write(',')
                    else:
                        result_file.write('\n')
                for i in range(len(acc_list)):
                    result = acc_list[i]
                    result_file.write(str(result))
                    if i != self.cfg.DATASET.communication_epoch - 1:
                        result_file.write(',')
                    else:
                        result_file.write('\n')

    # Write all results of all accounts
    #
    # all_path = all paths
    # al_acc_list - list of all accounts
    def write_all_acc(self, all_path, all_acc_list):
        if os.path.exists(all_path):
            with open(all_path, 'a') as result_file:
                for key in all_acc_list:
                    method_result = all_acc_list[key]
                    result_file.write(key + ',')
                    for epoch in range(len(method_result)):
                        result_file.write(str(method_result[epoch]))
                        if epoch != len(method_result) - 1:
                            result_file.write(',')
                        else:
                            result_file.write('\n')
        else:
            with open(all_path, 'w') as result_file:
                result_file.write('domain,')
                for epoch in range(self.cfg.DATASET.communication_epoch):
                    result_file.write('epoch_' + str(epoch))
                    if epoch != self.cfg.DATASET.communication_epoch - 1:
                        result_file.write(',')
                    else:
                        result_file.write('\n')

                for key in all_acc_list:
                    method_result = all_acc_list[key]
                    result_file.write(key + ',')
                    for epoch in range(len(method_result)):
                        result_file.write(str(method_result[epoch]))
                        if epoch != len(method_result) - 1:
                            result_file.write(',')
                        else:
                            result_file.write('\n')
