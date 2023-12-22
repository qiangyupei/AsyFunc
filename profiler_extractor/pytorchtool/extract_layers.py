# coding=utf-8
import os
import torch

path = "./model_weight"
dir_name = "default"

def save_weight_by_layer(module, name="", depth=-1):

    child_list = list(module.named_children())

    if depth == 0 or len(child_list) == 0:
        torch.save(module, os.path.join(path, dir_name, name + ".pkl"))
    else:
        for child in child_list:
            save_weight_by_layer(child[1], child[0] if name=="" else name + "." + child[0], depth - 1)

def save_model(model, model_name, depth=-1):
    global dir_name
    dir_name = model_name + "_layers"

    if not os.path.exists(os.path.join(path, dir_name)):
        os.makedirs(os.path.join(path, dir_name))
    
    save_weight_by_layer(model, name="", depth=depth)