from argparse import Namespace
from abc import abstractmethod


class FederatedAggregation:
    """
    Federated Aggregation
    """
    NAME = None

    def __init__(self, args: Namespace) -> None:
        self.args = args

    @abstractmethod
    def weight_calculate(self, **kwargs):
        pass

    # Aggregates all the parts of the model
    #
    # kwargs - required: freq, online_clients_list, nets_list, global_net, except_part, global_only
    #        - optional: global_w, use_additional_net, additional_net_list, additional_freq
    def agg_parts(self, **kwargs):
        # How many times a parameter is a value
        freq = kwargs['freq']
        # Simulated, random
        online_clients_list = kwargs['online_clients_list']
        # Potentially randomized NN
        nets_list = kwargs['nets_list']
        # Global version of the nets_list
        global_net = kwargs['global_net']
        # Dictionary holding the global weights
        global_w = {}
        # List
        except_part = kwargs['except_part']
        # Whether it is loaded only to global_net or also to net
        global_only = kwargs['global_only']

        use_additional_net = False
        additional_net_list = None
        additional_freq = None
        if 'use_additional_net' in kwargs:
            use_additional_net = kwargs['use_additional_net']
            additional_net_list = kwargs['additional_net_list']
            additional_freq = kwargs['additional_freq']

        first = True
        for index, net_id in enumerate(online_clients_list):
            net = nets_list[net_id]
            net_para = net.state_dict()

            # Node used to track parameters
            used_net_para = {}
            for k, v in net_para.items():
                is_in = False
                for part_str_index in range(len(except_part)):
                    if except_part[part_str_index] in k:
                        is_in = True
                        break

                if not is_in:
                    used_net_para[k] = v

            if first:
                first = False
                for key in used_net_para:
                    global_w[key] = used_net_para[key] * freq[index]
            else:
                for key in used_net_para:
                    global_w[key] += used_net_para[key] * freq[index]

        if use_additional_net:
            for index, _ in enumerate(additional_net_list):
                net = additional_net_list[index]
                net_para = net.state_dict()

                used_net_para = {}
                for k, v in net_para.items():
                    is_in = False
                    for part_str_index in range(len(except_part)):
                        if except_part[part_str_index] in k:
                            is_in = True
                            break

                    if not is_in:
                        used_net_para[k] = v

                for key in used_net_para:
                    global_w[key] += used_net_para[key] * additional_freq[index]

        if not global_only:
            for net in nets_list:
                net.load_state_dict(global_w, strict=False)

        global_net.load_state_dict(global_w, strict=False)
        return
