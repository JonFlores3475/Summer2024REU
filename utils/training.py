import copy

import torch.utils.data as data_utils
from Attack.byzantine.utils import attack_net_para
from Attack.Poisoning_Attack.utils import inverse_loss
from Methods.utils.meta_methods import FederatedMethod
from utils.logger import CsvWriter
import torch
import numpy as np
from utils.utils import log_msg
from utils.utils import row_into_parameters

import gradio as gr
from PIL import Image
import torchvision.transforms as T
import random
import pandas as pd
import time

global gallery
gallery = {"inputs":[], "outputs":{}}

gradio_interface = False

# if the dataset is cifar-10, change dataset_cifar to True, otherwise False
dataset_cifar = False
classes = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]

# This function is calculating the top1acc and top5acc (unsure what acc could be, maybe account?). This happens
# by the function getting the labels and finding the sum at a certain spot (so only 1 spot for the first one, and
# the whole list for the 5th one). At the end, it then multiplies those values by 100 and divides it by the total
# of the labels view.
#
# net - network potentially? whenever it's called, the variable being passed in is "global_net" so that makes me think
#       it's a type of network
# test_dl - appears to be some sort of domain list perhaps? 
# device - client that this is being calculated for
# return - returns the top1acc and the top5acc
def cal_top_one_five(net, test_dl, device):
    net.eval()
    correct, total, top1, top5 = 0.0, 0.0, 0.0, 0.0

    images_list = []
    labels_list = []

    # For the batch indexes, you get the images and labels at that index inside the test_dl 
    for batch_idx, (images, labels) in enumerate(test_dl):
        # Executes the next code if this doesn't fail
        with torch.no_grad():
            # Unsure what images and labels are being sit to
            images, labels = images.to(device), labels.to(device)
            # Gets the outputs of the nets images
            outputs = net(images)

            '''

            gets the classification for what the labels are being set as
            creates the interface
            
            '''

            if gradio_interface:
                if len(images_list) == 0:
                
                    transform_ = T.ToPILImage()
                    for image in images:
                        img = transform_(image)
                        images_list.append(img)
                    for label in labels:
                        labels_list.append(label)
                    
                    random_number = random.randint(0, len(images_list)-1)
                    input_image = images_list[random_number]
                    input_label = int(labels_list[random_number])

                    if dataset_cifar:
                        input_label = classes[input_label]

                    last_id = len(gallery["inputs"])

                    gallery["inputs"].append([input_image, "ID: "+str(last_id+1)]) # gr.Image(input_image, width=250, height=250, show_label=False)

                    last_id = len(gallery["inputs"])
                    gallery["outputs"][last_id] = input_label
                    
                    df = {
                        "ID" : [],
                        "Label": []
                    }

                    for id_ in gallery["outputs"]:
                        df["ID"].append(id_)
                        df["Label"].append(gallery["outputs"][id_])

                    def getLabel(inputs):
                        return gallery["inputs"]

                    #grvis = gr.Interface(fn=getLabel, inputs=gr.Image(input_image, width=250, height=250, show_label=False), outputs=gr.Label(input_label), live=True)
                    grvis = gr.Interface(fn=getLabel, inputs=[gr.Gallery(columns=[4], object_fit="contain", height="auto", value=gallery["inputs"], show_label=False)], outputs=[gr.Dataframe(pd.DataFrame(df), interactive=False)], live=True)
                    grvis.launch(server_port = 8080, prevent_thread_lock = True)

                    if len(gallery["inputs"]) % 10 == 0:
                        print("Inputs:",gallery["inputs"])
                        print("Outputs:",gallery["outputs"])
                        time.sleep(20)

                    if len(gallery["inputs"]) == 200:
                        cont = input("continue: ")
                    time.sleep(3)
                    grvis.close()

            # Gets the max5 variable from the torch.topk method, not caring about any of the other values
            _, max5 = torch.topk(outputs, 5, dim=-1)
            labels = labels.view(-1, 1)
            # Checks top1 to see if the labels equals max5's first column 
            top1 += (labels == max5[:, 0:1]).sum().item()
            # Checks top5 to see if the labels equals max5
            top5 += (labels == max5).sum().item()
            total += labels.size(0)
        # Retrains? Trains? the net
    net.train()
    # Calculates the top1acc and the top5acc and returns it
    top1acc = round(100 * top1 / total, 2)
    top5acc = round(100 * top5 / total, 2)
    return top1acc, top5acc

# Gets the in_domain_accs and the mean_in_domain_acc based on the federated method, test_loader dictionary, and a in_domain_list
# list. Training the global net after getting each top1acc when it is calculated.
#
# optimizer - federated method that is being used
# test_loader - dictionary of the test_loaders
# in_domain_list - list of the in_domains
# return - returns the in_domain_accs and the mean_in_domain_acc
def global_in_evaluation(optimizer: FederatedMethod, test_loader: dict, in_domain_list: list):
    in_domain_accs = []
    # For the in_domains in the in_domain_list
    for in_domain in in_domain_list:
        # If the optimizer has the attribute 'global_net_dict', it sets the global_net to the optimizers global_net_dict at
        # where the in_domain is.
        if hasattr(optimizer, 'global_net_dict'):
            global_net = optimizer.global_net_dict[in_domain]
        # If not, it sets the global net to the optimizers global_net
        else:
            global_net = optimizer.global_net
        global_net.eval()
        
        # Sets the test_domain_dl to the test_loader at the in_domain part of the dictionary 
        test_domain_dl = test_loader[in_domain]
        # Gets the top1acc (ignoring the top5acc)
        top1acc, _ = cal_top_one_five(net=global_net, test_dl=test_domain_dl, device=optimizer.device)
        # Appends the top1acc to the in_domain_accs
        in_domain_accs.append(top1acc)
        # Trains the global net
        global_net.train()
    # After the for loop finishes, it calculates the mean_in_domain_acc, returning both that and the in_domain_accs after
    mean_in_domain_acc = round(np.mean(in_domain_accs, axis=0), 3)
    return in_domain_accs, mean_in_domain_acc

# Calculates the sim_con_weight 
#
# kwargs - keyword arguments being passed in
# return - returns the sim_con_weight
def cal_sim_con_weight(**kwargs):
    # para
    optimizer = kwargs['optimizer']
    test_loader = kwargs['test_loader']
    task = kwargs['task']
    domain_list = kwargs['domain_list']

    # Creates local variable for organization
    global_net = optimizer.global_net
    # If the task is 'label_skew'. . .
    if task == 'label_skew':
        # Sets the overall_acc to the value of the cal_top_one_five value (top1_acc)
        overall_acc = cal_top_one_five(net=global_net, test_dl=test_loader, device=optimizer.device)
        overall_top1_acc = overall_acc[0]
    # Else, if the task is 'domain_skew'
    elif task == 'domain_skew':
        accs = []
        # For the in_domain in the domain_list . . .
        for in_domain in domain_list:
            # Sets the test_domain_dl to the test_loader's value at in_domain
            test_domain_dl = test_loader[in_domain]
            # Gets the top1acc from the cal_top_one_five() method
            top1acc, _ = cal_top_one_five(net=global_net, test_dl=test_domain_dl, device=optimizer.device)
            # Appends top1acc to the accs
            accs.append(top1acc)
        # After the for loop, it gets the overall_top1_acc by rounding the mean of accs
        overall_top1_acc = round(np.mean(accs, axis=0), 3)

    partial_acc_list = []
    # Creates a local variable for organization and simplicity
    nets_list = optimizer.nets_list_before_agg

    # If the optimizer has the attribute of 'aggregation_weight_list', it sets the aggregation_weight_list. If not, it prints
    # to the console saying that it doesn't support this method, exiting the program.
    if hasattr(optimizer, 'aggregation_weight_list'):
        aggregation_weight_list = optimizer.aggregation_weight_list
    else:
        print('not support this method')
        return

    # For the index_out in the online_clients_list
    for index_out, _ in enumerate(optimizer.online_clients_list):
        # Initializes a global_w (global weight) as well as many other variables
        global_w = {}
        temp_global_net = copy.deepcopy(global_net)
        temp_freq = copy.deepcopy(aggregation_weight_list)
        temp_freq[index_out] = 0
        temp_freq = temp_freq / np.sum(temp_freq)
        first = True
        # For the index and net_id in the online_clients_list
        for index, net_id in enumerate(optimizer.online_clients_list):
            # Sets the net to the net in nets_list at the net_id
            net = nets_list[net_id]
            # Sets the net_para to the net's state_dict
            net_para = net.state_dict()

            except_part = []
            used_net_para = {}
            # For the key and value in the net_parameters items
            for k, v in net_para.items():
                # Initially sets is_in to false
                is_in = False
                # For the part_str_index in the range of the length of the except_part
                for part_str_index in range(len(except_part)):
                    # If the except_part at part_str_index is in k (key), set is_in to True and break the inner loop
                    if except_part[part_str_index] in k:
                        is_in = True
                        break
                # If is_in is False, set the used_net_para at k (key) to v (value)
                if not is_in:
                    used_net_para[k] = v

            # If first is True
            if first:
                # Set first to False
                first = False
                # For the keys in the used_net_para, set the global_w at the key index to the product of the used_net_para
                # at the certain key and the temp_freq at a certain index
                for key in used_net_para:
                    global_w[key] = used_net_para[key] * temp_freq[index]
            # If first is False
            else:
                # For the keys in the used_net_para, set (and continuously add to) the global_w at the key index to the product
                # of the used_net_para at a certain key and the temp_freq at a certain index
                for key in used_net_para:
                    global_w[key] += used_net_para[key] * temp_freq[index]
        # For the temporary global net, load the state dictionary with the global weight and setting strict to False
        temp_global_net.load_state_dict(global_w, strict=False)

        # If the task is 'label_skew', it sets the partial_top1_acc and partial_top5_acc to the values returned by the cal_top_one_five()
        if task == 'label_skew':
            partial_top1_acc, partial_top5_acc = cal_top_one_five(net=temp_global_net, test_dl=test_loader, device=optimizer.device)
        # Else, if the task is 'domain_skew' . . .
        elif task == 'domain_skew':
            accs = []
            # For the in_domain in the domain_list
            for in_domain in domain_list:
                # Set the test_domain_dl to the test_loader at the in_domain index
                test_domain_dl = test_loader[in_domain]
                # Get the top1acc from the cal_top_one_five
                top1acc, _ = cal_top_one_five(net=temp_global_net, test_dl=test_domain_dl, device=optimizer.device)
                # Append the top1acc to the accs
                accs.append(top1acc)
            # After the for loop, set the partial_top_1_acc to the rounded mean of the accs
            partial_top1_acc = round(np.mean(accs, axis=0), 3)
        # Append the partial_top1_acc to the partial_acc_list
        partial_acc_list.append(partial_top1_acc)

    # Get the overall_top1_acc_list t the list of overall_top1_acc multiplied by the length of the partial_acc_list
    overall_top1_acc_list = [overall_top1_acc] * len(partial_acc_list)
    # Sets dif_ac
    dif_ac = [a - b + 1e-5 for a, b in zip(overall_top1_acc_list, partial_acc_list)]
    dif_ac = dif_ac / (np.sum(dif_ac))
    print(partial_acc_list)
    # Calculates the sim_con_weight and returns it
    sim_con_weight = dif_ac.dot(aggregation_weight_list) / (
            np.linalg.norm(dif_ac) * np.linalg.norm(aggregation_weight_list))
    return sim_con_weight

# Gets the out_acc (top1acc) based on a federated method, specific test loader, and domain list, training the global_net
# after it has calculated the out_acc by calling the cal_top_one_five() method.
#
# optimizer - the federated method being used (i.e. DelphiflMedian)
# test_loader - test_loader dictionary
# out_domain - string of a domain_list (I think?)
# return - returns the out_acc
def global_out_evaluation(optimizer: FederatedMethod, test_loader: dict, out_domain: str):
    test_out_domain_dl = test_loader[out_domain]

    # If the optimizer has the attribute of 'global_net_dict' . . .
    if hasattr(optimizer, 'global_net_dict'):
        global_net = optimizer.global_net_dict[out_domain]
    else:
        global_net = optimizer.global_net
    global_net.eval()
    # Gets the top1acc, ignoring the top5acc
    top1acc, _ = cal_top_one_five(net=global_net, test_dl=test_out_domain_dl, device=optimizer.device)
    # Sets the out_acc to the value of top1acc
    out_acc = top1acc
    # Trains the net afterwards and returns the out_acc
    global_net.train()
    return out_acc

# Sets the values of the keys inside the dictionary to a 0 instead of a blank.
#
# net_cls_counts - the net_cls_counts from a private_dataset
# classes - a list of the classes that are inside the cfg's dataset
# return - returns the net_cls_counts
def fill_blank(net_cls_counts,classes):
    class1 = [i for i in range(classes)]
    # Gets the client and the dict_i from the net_cls_counts items
    for client, dict_i in net_cls_counts.items():
        # If the dictionary's length is 10, continue to the next dictionary
        if len(dict_i.keys()) == 10:
            continue
        else:
        # If it isn't, go through i from class1
            for i in class1:
                # If i isn't in the dictionary's keys, add it and set the value of the key to 0.
                if i not in dict_i.keys():
                    dict_i[i] = 0
    # Return the updated net_cls_counts at the end
    return net_cls_counts

# Trains the model based off a bunch of different rules and parameters.
#
# fed_method - method the FL is using
# private_dataset - dataset the model is using
# args - arguments coming from main (flags)
# cfg - CfgNode being used (local model?)
# client_domain_list - list of client indexes
def train(fed_method, private_dataset, args, cfg, client_domain_list, client_type) -> None:
    # Checks to see if we are logging the steps, creating a CsvWriter if we are
    if args.csv_log:
        csv_writer = CsvWriter(args, cfg)

    # Checks to see if the federated method has the attribute 'ini', executing it if it does
    if hasattr(fed_method, 'ini'):
        fed_method.ini()

    # Checks to too if the arguments' task is 'OOD', creating specific settings based off it
    if args.task == 'OOD':
        in_domain_accs_dict = {}  # Query-Client Accuracy A^u
        mean_in_domain_acc_list = []  # Cross-Client Accuracy A^U
        out_domain_accs_dict = {}  # Out-Client Accuracy A^o
        fed_method.out_train_loader = private_dataset.out_train_loader
    # Else, if the arguments' task is 'label_skew', it creates the specific settings for that task
    elif args.task == 'label_skew':
        mean_in_domain_acc_list = []
        if args.attack_type == 'None':
            contribution_match_degree_list = []
        # Fills the rest of the net_cls_counts to 0's if they are blank
        fed_method.net_cls_counts = fill_blank(private_dataset.net_cls_counts,cfg.DATASET.n_classes)
    # Else, if the arguments' task is 'domain_skew', it creates the specific settings for that task
    elif args.task == 'domain_skew':
        in_domain_accs_dict = {}  # Query-Client Accuracy \bm{\mathcal{A}}}^{u}
        mean_in_domain_acc_list = []  # Cross-Client Accuracy A^U \bm{\mathcal{A}}}^{\mathcal{U}
        performance_variane_list = []
        if args.attack_type == 'None':
            contribution_match_degree_list = []
    # If the arguments' attack_type is backdoor, it creates a list for the attack_success_rate
    if args.attack_type == 'backdoor' or (args.attack_type == 'Poisoning_Attack' and args.poisoning_evils == 'inverted_gradient' and args.backdoor_evils == 'atropos'):
        attack_success_rate = []
            
    # Creates a local variable for organization of the communication_epoch
    communication_epoch = cfg.DATASET.communication_epoch
    # For each epoch in the communication_epoch range
    for epoch_index in range(communication_epoch):
        # Set the federated methods variables
        losses = []
        fed_method.epoch_index = epoch_index

        # Client
        fed_method.test_loader = private_dataset.test_loader
        # Locally updates
        if args.attack_type == "Poisoning_Attack":
            for client_index in range(cfg.DATASET.parti_num):
                if not client_type[client_index]:
                    convert, remove = next(iter(private_dataset.test_loader))
                    train, remove = next(iter(private_dataset.train_loaders[client_index]))
                    loss = inverse_loss(train, convert)
                    losses.append(loss)
                else:
                    loss = -1
                    losses.append(loss)
            fed_method.local_update(private_dataset.train_loaders, losses)
                # row_into_parameters(loss, np.array(private_dataset.train_loaders[0]))
                # train_loader.append(data_utils.DataLoader(loss, batch_size=len(private_dataset.train_loaders[0]), shuffle=True))
        else:
            fed_method.local_update(private_dataset.train_loaders)

        fed_method.nets_list_before_agg = copy.deepcopy(fed_method.nets_list)

        # If the arguments' attack_type is 'byzantine', calls a method that creates the attack net parameters
        if args.attack_type == 'byzantine':
            attack_net_para(args, cfg, fed_method)

        # Server
        fed_method.sever_update(private_dataset.train_loaders)
        print("test1")
        # If that arguments' task is 'OOD'
        if args.task == 'OOD':
            '''
            domain_accs
            '''
            # Checks to see if the federated method has the attribute 'weight_dict'
            if hasattr(fed_method, 'weight_dict'):
                # Creates a local variable for organization
                    weight_dict = fed_method.weight_dict
                # If we are logging, write the weight into the csv file
                    if args.csv_log:
                        csv_writer.write_weight(weight_dict, epoch_index, client_domain_list)
            # Sets the domain_accs and the mean_in_domain_acc 
            domain_accs, mean_in_domain_acc = global_in_evaluation(fed_method, private_dataset.test_loader, private_dataset.in_domain_list)
            # Appends the mean_in_domain_acc to the current mean_in_domain_acc_list
            mean_in_domain_acc_list.append(mean_in_domain_acc)
            # For the index and in_domain in the private_dataset in_domain_list
            for index, in_domain in enumerate(private_dataset.in_domain_list):
                # If the in_domain is in the in_domain_accs_dict, appends the domains_accs at a certain index to the
                # in_domain_accs_dict at the spot of in_domain
                if in_domain in in_domain_accs_dict:
                    in_domain_accs_dict[in_domain].append(domain_accs[index])
                else:
                    # If it isn't, just set it equal to it instead of appending
                    in_domain_accs_dict[in_domain] = [domain_accs[index]]
            # Prints a log message
            print(log_msg(f"The {epoch_index} Epoch: In Domain Mean Acc: {mean_in_domain_acc} Method: {args.method} CSV: {args.csv_name}", "TEST"))
            '''
            OOD
            '''
            # If the out_domain isn't 'NONE'
            if cfg[args.task].out_domain != "NONE":
                # Sets the out_domain_acc
                out_domain_acc = global_out_evaluation(fed_method, private_dataset.test_loader, cfg[args.task].out_domain)
                # If the out_domain of the cfg[args.task] is in the out_domain_acs_dict, append the out_domain_acc
                if cfg[args.task].out_domain in out_domain_accs_dict:
                    out_domain_accs_dict[cfg[args.task].out_domain].append(out_domain_acc)
                else:
                    # If not, just set it equal to it instead of appending 
                    out_domain_accs_dict[cfg[args.task].out_domain] = [out_domain_acc]
                # Prints a log message
                print(log_msg(f"The {epoch_index} Epoch: Out Domain {cfg[args.task].out_domain} Acc: {out_domain_acc} Method: {args.method} CSV: {args.csv_name}", "OOD"))
        # Else, if the arguments' task is NOT 'OOD'
        else:
        # If the 'mean_in_domain_acc_list' is in the locals and the arguments' task is "label_skew"
            print("test2")
            if 'mean_in_domain_acc_list' in locals() and args.task == 'label_skew':
                print("eval mean_in_domain_acc_list -- test3")
                # Gets the top1acc from the cal_top_one_five method, appending it to the mean_in_domain_acc_list
                top1acc, _ = cal_top_one_five(fed_method.global_net, private_dataset.test_loader, fed_method.device)
                mean_in_domain_acc_list.append(top1acc)
                print(log_msg(f'The {epoch_index} Epoch: Acc:{top1acc}', "TEST"))

        # If the 'contribution_match_degree_list' is in locals and the federated method aggregation_weight_list is not none. . .
            if 'contribution_match_degree_list' in locals() and fed_method.aggregation_weight_list is not None:
                print("eval contribution_match_degree_list")
                # Checks to see if the epoch index is divisible by 10, OR if the epoch index is one less than the communication_epoch
                if epoch_index % 10 == 0 or epoch_index == communication_epoch - 1:
                    # If so, checks to see if the arguments' task is 'label_skew'. If so, sets the domain_list to None
                    if args.task == 'label_skew':
                        domain_list = None
                    # Else, if the arguments' task is 'domain_skew, sets the domain_list to the private datasets' domain_list
                    elif args.task == 'domain_skew':
                        domain_list = private_dataset.domain_list
                    # The con_fair_metric is set to the cal_sim_con_wright() method
                    con_fair_metric = cal_sim_con_weight(optimizer=fed_method, test_loader=private_dataset.test_loader,
                                                        domain_list=domain_list, task=args.task)
                    # Appends the con_fair_metric to th contribution_match_degree_list
                    contribution_match_degree_list.append(con_fair_metric)
                # If it isn't the case (line 361)
                else:
                    # Sets the con_fair_metric to 0 and appends it to the contribution_match_degree_list
                    con_fair_metric = 0
                    contribution_match_degree_list.append(con_fair_metric)
                print(log_msg(f'The {epoch_index} Method: {args.method} Epoch: Con Fair:{con_fair_metric}', "TEST"))

            # If 'in_domain_accs_dict' is in the locals
            if 'in_domain_accs_dict' in locals():
                print("eval in_domain_accs_dict")
                # Sets the domain_accs and the mean_in_domain_acc to the values returned by the global_in_evaluation()
                domain_accs, mean_in_domain_acc = global_in_evaluation(fed_method, private_dataset.test_loader, private_dataset.domain_list)                    # Sets the performance variable and appends it to the performance_variane_list
                perf_var = np.var(domain_accs, ddof=0)
                performance_variane_list.append(perf_var)
                mean_in_domain_acc_list.append(mean_in_domain_acc)

                # For the index and the in_domain inside the private_datasets domain_list
                for index, in_domain in enumerate(private_dataset.domain_list):
                # If the in_domain is in the in_domain_accs_dict, it appends the domain_accs at a certain index to the                        # in_domain_accs_dict at the in_domain key
                    if in_domain in in_domain_accs_dict:
                        in_domain_accs_dict[in_domain].append(domain_accs[index])
                        # If not, then it sets it instead of appending it
                    else:
                        in_domain_accs_dict[in_domain] = [domain_accs[index]]
                    print(log_msg(f"The {epoch_index} Epoch: Mean Acc: {mean_in_domain_acc} Method: {args.method} Per Var: {perf_var} ", "TEST"))

            # If 'attack_success_rate is in locals
            if 'attack_success_rate' in locals():
                # Gets the top1acc from the cal_top_one_five()
                top1acc, _ = cal_top_one_five(fed_method.global_net, private_dataset.backdoor_test_loader, fed_method.device)
                # Appends the top1acc to the attack_success_rate
                attack_success_rate.append(top1acc)
                print(log_msg(f'The {epoch_index} Epoch: attack success rate:{top1acc}'))

    # If we are logging to a csv_file
    if args.csv_log:
        
        # If the arguments' task is 'OOD'
        if args.task == 'OOD':
            # Writes the mean_in_domain_acc_list, in_domain_accs_dict to the csv file.
            csv_writer.write_acc(mean_in_domain_acc_list, name='in_domain', mode='MEAN')
            csv_writer.write_acc(in_domain_accs_dict, name='in_domain', mode='ALL')
            # If the out_domain at the cfg args' task key is not 'NONE', it writes out_domain_accs_dict to the csv file
            if cfg[args.task].out_domain != "NONE":
                csv_writer.write_acc(out_domain_accs_dict, name='out_domain', mode='ALL')

        # Else, if the arguments' task is 'label_skew'
        elif args.task == 'label_skew':
            # Write the mean_in_domain_acc_list to the csv file
            csv_writer.write_acc(mean_in_domain_acc_list, name='in_domain', mode='MEAN')
            # If the arguments' attack type is none, writes the contribution_match_degree_list to the csv file
            if args.attack_type == 'None':
                csv_writer.write_acc(contribution_match_degree_list, name='contribution_fairness', mode='MEAN')

        # Else, if the arguments' task is 'domain_skew'
        elif args.task == 'domain_skew':
            # Writes the mean_in_domain_acc_list, in_domain_accs_dict, contribution_match_degree_list, and the performance_varian_list
            # to the csv file
            csv_writer.write_acc(mean_in_domain_acc_list, name='in_domain', mode='MEAN')
            csv_writer.write_acc(in_domain_accs_dict, name='in_domain', mode='ALL')
            csv_writer.write_acc(contribution_match_degree_list, name='contribution_fairness', mode='MEAN')
            csv_writer.write_acc(performance_variane_list, name='performance_variance', mode='MEAN')
        # If the arguments' attack_type is 'backdoor', it writes the attack_success_rate to the csv file
        if args.attack_type == 'backdoor' or (args.attack_type == 'Poisoning_Attack' and args.poisoning_evils == 'inverted_gradient' and args.backdoor_evils == 'atropos'):
            csv_writer.write_acc(attack_success_rate, name='attack_success_rate', mode='MEAN')

