'''
import copy

import numpy as np
import keras
import torch
import tensorflow as tf
from torch.utils.data import DataLoader

# --------------------------------
# import sympy 
from sympy import *
 
x, y = symbols('x y')
expr = x**2 + 2 * y + y**3
print("Expression : {}".format(expr))
  
# Use sympy.Derivative() method 
expr_diff = Derivative(expr, x)  
     
print("Derivative of expression with respect to x : {}".format(expr_diff))  
print("Value of the derivative : {}".format(expr_diff.doit()))
# --------------------------------

# --------------------------------
# Sneaky randomness (attempt at hiding in Gaussian noise) (generally messing things up)
# Found out that img may be the weights and not the image itself
# These are all attempts at accomplishing roughly the same thing
# Randomly shuffles the image, but only a percentage of them that are less than the noise rate
def sneaky_random(img, noise_data_rate):
    if torch.rand(1) < noise_data_rate:
        img = img * (1 + torch.randn(img.size()))
    return img

# Randomly shuffles the image, but making sure that the amount shuffled is less than or equal to the noise rate
def sneaky_random2(img, noise_data_rate):
    randTensor = torch.randn(img.size())
    for x in randTensor:
        if x > noise_data_rate:
            x = noise_data_rate
    img = img * (1 + randTensor)
    return img

# Randomly shuffles the image and the target, but only a percentage of them that are less than the noise rate
def sneaky_random3(img, target, noise_data_rate):
    if torch.rand(1) < noise_data_rate:
        img = img * (1 + torch.randn(img.size()))
        target = int(torch.rand(1) * 10)
    return img, target

# Randomly shuffles the target, but only a percentage of them that are less than the noise rate
def sneaky_random4(target, noise_data_rate):
    if torch.rand(1) < noise_data_rate:
        target = int(torch.rand(1) * 10)
    return target

# Randomly shuffles the image, with the amount shuffled being an addition rather than a multiplication
# and actually shuffled how I wanted to do so last time. Probably less efficient
# UNTESTED
def sneaky_random5(img, noise_data_rate):
    randTensor = torch.randn(img.size())
    print(randTensor.shape)
    width, height, depth = randTensor.shape
    print(width)
    print(height)
    print(depth)
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                if randTensor[x][y][z] > noise_data_rate:
                    randTensor[x][y][z] = noise_data_rate
    img = img + randTensor
    return img

# Generates Gaussian noise centered around 128 in the image given and randomly assigns a label
# Likely does not work as is (may need to import PIL)
# TODO: Check if 1 is the correct sigma and check if img.size works as intended (it should)
def gaus_images(img, target):
    img = img.effect_noise(img.size, 1)
    target = int(torch.rand(1) * 10)
    return img, target

# Resizes the image twice so that it is more blurry than before and randomly assigns a label
# Likely does not work as is (may need to import PIL)
# TODO: Check if the resize is being used properly, if .size can be used as is, and whether NEAREST or BOX is better
def shrink_stretch(img, target):
    img = img.resize(img.size / 2, Resampling.NEAREST) # - most efficient but worst
    img = img.resize(img.size * 2, Resampling.BOX) # - less efficient and second-worst
    target = int(torch.rand(1) * 10)
    return img, target
# --------------------------------

def inverse_loss(target, prediction):
    loss = keras.losses.categorical_crossentropy(target, prediction)
    inv_loss = torch.zeros(tuple(loss.shape))
    start = 0
    for i in range(loss.shape[0]):
        for j in range(loss.shape[1]):
            for k in range(loss.shape[2]):
                start+=tf.get_static_value(loss[i,j,k])
                if start < 0.001:
                    inv_loss[i,j,k] = 1/0.001
                else:
                    inv_loss[i,j,k] = 1 / start
                start = 0
    return inv_loss

# Inverted Gradient Attack
def inverted_gradient(args, cfg, client_type, private_dataset, is_train):
    num_nodes = cfg.attack.bad_client_rate * cfg.DATASET.parti_num # N
    y = copy.deepcopy(private_dataset.train_loaders) # y^ ??
    y_hat = copy.deepcopy(private_dataset.out_train_loaders)
    # img, target = dataset.__getitem__(0) # xi and yi
    # cfg.attack.noise_data_rate # alpha
    # weight = lambda_vector.cpu().np() # w


# Base backdoor method is a more secure backdoor that is (potentially) used for more
# stealth in a backdoor attack, but it isn't as detrimental.
# This sets the target to the cfg's backdoor label, and then for every position in the
# trigger_positions list, it sets that position to said trigger_position and updates part of the image.
# This makes it so the part of the image contains a backdoor that can be used later.
#
# cfg - the CFG node 
# img - the image (model) being attacked
# target - the type of target attack being used (think of labels)
# noise_data_rate - the noise data rate of the CFG node
def base_backdoor(cfg, img, target, noise_data_rate):
    if torch.rand(1) < noise_data_rate: # Erin: Is this just a randomizer?
        target = cfg.attack.backdoor.backdoor_label
        for pos_index in range(0, len(cfg.attack.backdoor.trigger_position)):
            pos = cfg.attack.backdoor.trigger_position[pos_index]
            img[pos[0]][pos[1]][pos[2]] = cfg.attack.backdoor.trigger_value[pos_index]
    return img, target

# Semantic backdoor checks a random number from torch to see if it is a semantic backdoor.
# if that condition passes it sets the target to a backdoor_label IFF the target is currently
# a semantic_backdoor_label. After that, the img is then altered to have more space for the label,
# including a semantic_backdoor into the image. This attack is more detrimental, yet more noticeable.
#
# cfg - the CFG node 
# img - the image (model) being attacked
# target - the type of target attack being used (think of labels)
# noise_data_rate - the noise data rate of the CFG node
def semantic_backdoor(cfg, img, target, noise_data_rate):
    if torch.rand(1) < noise_data_rate: # Erin: Is this just a randomizer?
        if target == cfg.attack.backdoor.semantic_backdoor_label:
            target = cfg.attack.backdoor.backdoor_label

            # img, _ = dataset.__getitem__(used_index)
            img = img + torch.randn(img.size()) * 0.05
    return img, target

# This function is the actual attack of the backdoor itself. 
#
# args - the arguments passed in through main
# cfg - the CFG node being passed in
# client_type - a list of booleans saying if it is being backdoored or not (true if backdoored, false if not)
# private_dataset - the private dataset of the FL system 
# is_train - a boolean saying if the system is in the training phase
def backdoor_attack(args, cfg, client_type, private_dataset, is_train):
    # Gets the noise_data_rate of the cfg iff it is in the training stage, setting it to 1.0 if it isn't
    noise_data_rate = cfg.attack.noise_data_rate if is_train else 1.0
    # Checks to see if it is in the training stage
    if is_train:
        # For every client index in the range of the cfg's dataset participant numbers
        for client_index in range(cfg.DATASET.parti_num):
            # If the client_type at the index is false (so not backdoored) . . .
            if not client_type[client_index]:
                # Creates a deepcopy of the private_dataset
                dataset = copy.deepcopy(private_dataset.train_loaders[client_index].dataset)

                all_targets = []
                all_imgs = []
                # For i in the range of the length of the dataset dictionary
                for i in range(len(dataset)):
                    # Gets the original image (model) and target
                    img, target = dataset.__getitem__(i)
                    # Checks to see if the backdoor is a base_backdoor
                    if cfg.attack.backdoor.evils == 'base_backdoor':
                        # If so, sets the img and target to the results of the base_backdoor attack method
                        img, target = base_backdoor(cfg, copy.deepcopy(img), copy.deepcopy(target), noise_data_rate)
                    # If not, checks to see if the backdoor is a semantic_backdoor
                    elif cfg.attack.backdoor.evils == 'semantic_backdoor':
                        # If so, sets the img and target to the results of the semantic_backdoor attack method
                        img, target = semantic_backdoor(cfg, copy.deepcopy(img), copy.deepcopy(target), noise_data_rate)
                    # If neither, prints an error message
                    else:
                        print("ERROR: Choose between base backdoor and semantic backdoor")
                    # Adds the target and image (model) to their respective all_* lists
                    all_targets.append(target)
                    all_imgs.append(img.numpy())
                # Sets the new_dataset to a BackdoorDataset with the new images and targets
                new_dataset = BackdoorDataset(all_imgs, all_targets)
                # Gets the sampler of the training stage
                train_sampler = private_dataset.train_loaders[client_index].batch_sampler.sampler
                # If the task of the dataset is to label_skew, it sets the train_loaders at the certain client index to their own parameters
                if args.task == 'label_skew':
                    private_dataset.train_loaders[client_index] = DataLoader(new_dataset, batch_size=cfg.OPTIMIZER.local_train_batch,
                                                                             sampler=train_sampler, num_workers=4, drop_last=True)
                else:
                    # Is this necessary?
                    print("--task is not equal to label_skew")
    # If it isn't in the training stage
    else:
        # Checks to see if the task is label_skew . . .
        if args.task == 'label_skew':
            # If so, it creates a deepcopy of the private_dataset
            dataset = copy.deepcopy(private_dataset.test_loader.dataset)

            all_targets = []
            all_imgs = []
            
            # For i in the range of the length of the dataset dictionary
            for i in range(len(dataset)):
                # Gets the original image (model) and target
                img, target = dataset.__getitem__(i)
                # Checks to see if the attack is a type of base_backdoor
                if cfg.attack.backdoor.evils == 'base_backdoor':
                    # If so, it does the base_backdoor attack
                    img, target = base_backdoor(cfg, copy.deepcopy(img), copy.deepcopy(target), noise_data_rate)
                    # It then appends the target and image (model) to their own respective lists 
                    all_targets.append(target)
                    all_imgs.append(img.numpy())
                # If not, then it checks to see if the attack is a type of semantic_backdoor
                elif cfg.attack.backdoor.evils == 'semantic_backdoor':
                    # Checks to see if the target has a semantic_backdoor_label
                    if target == cfg.attack.backdoor.semantic_backdoor_label:
                        # If it does, then it executes a semantic_backdoor attack
                        img, target = semantic_backdoor(cfg, copy.deepcopy(img), copy.deepcopy(target), noise_data_rate)
                        # It then appends the target and image(model) to their own respective lists
                        all_targets.append(target)
                        all_imgs.append(img.numpy())
                # Prints an error message if neither of the conditions above pass
                else:
                    print("ERROR: Choose between base backdoor and semantic backdoor")

                # all_targets.append(target)
                # all_imgs.append(img.numpy())
            # Creates a new dataset with the BackdoorDataset information (getting from all_*)
            new_dataset = BackdoorDataset(all_imgs, all_targets)
            # It then sets the private_datasets backdoor_test_loader to a new DataLoader based on their own parameters
            private_dataset.backdoor_test_loader = DataLoader(new_dataset, batch_size=cfg.OPTIMIZER.local_train_batch, num_workers=4)
        # Prints an error statement if neither of them pass 
        else:
            print("ERROR: --task should be set to label_skew in order to run the backdoor attack without is_train")


class BackdoorDataset(torch.utils.data.Dataset):

    def __init__(self, data, labels):
        self.data = np.array(data)
        self.labels = np.array(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]
'''
