import torch

from Local.utils.local_methods import LocalMethod
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm


class FedProxLocal(LocalMethod):
    NAME = 'FedProxLocal'

    def __init__(self, args, cfg):
        super(FedProxLocal, self).__init__(args, cfg)
        self.mu = cfg.Local[self.NAME].mu

    # Local update, calls train_net
    #
    # kwargs - online_clients_list, nets_list, priloader_list, global_net
    def loc_update(self, loss, **kwargs):
        # Simulated and randomized, local version of the online_clients list in meta_aggregation.py
        online_clients_list = kwargs['online_clients_list']
        # Potential randomized neural network, local version of the nets_list in meta_aggregation.py
        nets_list = kwargs['nets_list']
        # List, called as train_loader
        priloader_list = kwargs['priloader_list']
        # Global neural network from nets_list
        global_net = kwargs['global_net']
        for i in online_clients_list:
            self.train_net(i, nets_list[i], global_net, priloader_list[i], loss)

    # Trains the neural network
    #
    # index - the index of the online_clients_list
    # net - specific net to be trained
    # global_net - global version of the net
    # train_loader - specific value in the priloader_list
    def train_net(self, index, net, global_net, train_loader, initial_losses):
        net = net.to(self.device)
        net.train()
        if self.cfg.OPTIMIZER.type == 'SGD':
            optimizer = optim.SGD(net.parameters(), lr=self.cfg.OPTIMIZER.local_train_lr,
                                  momentum=self.cfg.OPTIMIZER.momentum, weight_decay=self.cfg.OPTIMIZER.weight_decay)
        criterion = nn.CrossEntropyLoss()
        
        criterion.to(self.device)
        iterator = tqdm(range(self.cfg.OPTIMIZER.local_epoch))
        global_net = global_net.to(self.device)
        global_weight_collector = list(global_net.parameters())
        # criterion = criterion.numpy()
        for _ in iterator:
            for batch_idx, (images, labels) in enumerate(train_loader):
                for loss_idx in len(initial_losses):
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    outputs = net(images)
                    if initial_losses[loss_idx] == -1:
                        loss = criterion(outputs, labels)
                    else:
                        loss = initial_losses[loss_idx]
                    fed_prox_reg = 0.0
                    for param_index, param in enumerate(net.parameters()):
                        fed_prox_reg += ((0.01 / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)
                    loss += self.mu * fed_prox_reg
                    optimizer.zero_grad()
                    loss.backward()
                    iterator.desc = "Local Participant %d loss = %0.3f" % (index, loss)
                    optimizer.step()
