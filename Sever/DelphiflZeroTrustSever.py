import copy

import numpy as np
import torch
from torch import optim, nn
from tqdm import tqdm

from Backbones import get_private_backbones
from Datasets.public_dataset import get_public_dataset
from Sever.utils.utils import DelphiflMedian
from Sever.utils.sever_methods import SeverMethod

from utils.utils import row_into_parameters

class DelphiflZeroTrustSever(SeverMethod):
    NAME = 'DelphiflZeroTrustSever'

    def __init__(self, args, cfg):
        super(DelphiflZeroTrustSever, self).__init__(args, cfg)
        
        nets_list = get_private_backbones(cfg)


        public_dataset_name = cfg.Sever[self.NAME].public_dataset_name
        pub_len = cfg.Sever[self.NAME].pub_len
        pub_aug = cfg.Sever[self.NAME].pub_aug
        public_batch_size = cfg.Sever[self.NAME].public_batch_size
        self.public_epoch = cfg.Sever[self.NAME].public_epoch
        self.public_dataset = get_public_dataset(args, cfg, public_dataset_name=public_dataset_name,
                                                 pub_len=pub_len, pub_aug=pub_aug, public_batch_size=public_batch_size)
        self.public_dataset.get_data_loaders()
        self.public_loader = self.public_dataset.traindl
        
        self.momentum = 0.9
        self.learning_rate = self.cfg.OPTIMIZER.local_train_lr
        self.current_weights = []
        for name, param in copy.deepcopy(nets_list[0]).cpu().state_dict().items():
            param = nets_list[0].state_dict()[name].view(-1)
            self.current_weights.append(param)
        self.current_weights = torch.cat(self.current_weights, dim=0).cpu().numpy()
        self.velocity = np.zeros(self.current_weights.shape, self.current_weights.dtype)
        self.n = 5

    def sever_update(self, **kwargs):

        online_clients_list = kwargs['online_clients_list']

        global_net = kwargs['global_net']
        nets_list = kwargs['nets_list']
        temp_net = copy.deepcopy(global_net)

        with torch.no_grad():
            all_delta = []
            global_net_para = []
            add_global = True
            for i in online_clients_list:

                net_all_delta = []
                for name, param0 in temp_net.state_dict().items():
                    param1 = nets_list[i].state_dict()[name]
                    delta = (param1.detach() - param0.detach())

                    net_all_delta.append(copy.deepcopy(delta.view(-1)))
                    if add_global:
                        weights = copy.deepcopy(param0.detach().view(-1))
                        global_net_para.append(weights)

                add_global = False
                net_all_delta = torch.cat(net_all_delta, dim=0).cpu().numpy()
                all_delta.append(net_all_delta)

            all_delta = np.array(all_delta)
            global_net_para = np.array(torch.cat(global_net_para, dim=0).cpu().numpy())

        criterion = nn.CrossEntropyLoss()
        iterator = tqdm(range(self.public_epoch))
        optimizer = optim.SGD(temp_net.parameters(), lr=self.cfg.OPTIMIZER.local_train_lr,
                              momentum=self.cfg.OPTIMIZER.momentum, weight_decay=self.cfg.OPTIMIZER.weight_decay)
        for _ in iterator:
            for batch_idx, (images, labels) in enumerate(self.public_loader):
                images = images
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = temp_net(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            global_delta = []
            for name, param0 in temp_net.state_dict().items():
                param1 = global_net.state_dict()[name]
                delta = (param0.detach() - param1.detach())
                global_delta.append(copy.deepcopy(delta.view(-1)))

            global_delta = torch.cat(global_delta, dim=0).cpu().numpy()
            global_delta = np.array(global_delta)

        total_TS = 0
        TSnorm = []
        for d in all_delta:
            tmp_weight = copy.deepcopy(d)

            TS = np.dot(tmp_weight, global_delta) / (np.linalg.norm(tmp_weight) * np.linalg.norm(global_delta) + 1e-5)
            # print(TS)
            if TS < 0:
                TS = 0
            total_TS += TS

            norm = np.linalg.norm(global_delta) / (np.linalg.norm(tmp_weight) + 1e-5)
            TSnorm.append(TS * norm)
        
        delta_weight = np.sum(np.array(TSnorm).reshape(-1, 1) * all_delta, axis=0) / (total_TS + 1e-5)

                
        with torch.no_grad():
            temp_net = copy.deepcopy(global_net)
            all_grads = []
            for i in online_clients_list:
                grads = {}
                net_all_grads = []
                for name, param0 in temp_net.state_dict().items():
                    param1 = nets_list[i].state_dict()[name]
                    grads[name] = (param0.detach() - param1.detach()) / self.learning_rate
                    net_all_grads.append(copy.deepcopy(grads[name].view(-1)))

                net_all_grads = torch.cat(net_all_grads, dim=0).cpu().numpy()
                all_grads.append(net_all_grads)
            all_grads = np.array(all_grads)

        # bad_client_num = int(self.args.bad_client_rate * len(self.online_clients))
        f = len(online_clients_list) // 2  # worse case 50% malicious points
        k = len(online_clients_list) - f - 1

        current_grads = DelphiflMedian(all_grads, len(online_clients_list), f - k, n=self.n)

        self.velocity = self.velocity * delta_weight - self.learning_rate * current_grads
        self.current_weights += self.velocity

        row_into_parameters(self.current_weights, global_net.parameters())
        for _, net in enumerate(nets_list):
            net.load_state_dict(global_net.state_dict())
