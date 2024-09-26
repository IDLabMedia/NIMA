"""
file - test.py
Simple quick script to evaluate model on test images.

Copyright (C) Yunxiao Shi 2017 - 2021
NIMA is released under the MIT license. See LICENSE for the fill license text.
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from tqdm import tqdm
import torch
import torchvision.models as models
import torchvision.transforms as transforms

from .model.model import *

def load_model(model_path, vgg16_model_path=None):
    if vgg16_model_path == None:
        base_model = models.vgg16(pretrained=True)
    else:
        base_model = models.vgg16()
        base_model.load_state_dict(torch.load(vgg16_model_path))
    model = NIMA(base_model)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print('successfully loaded model')
    except:
        raise

    seed = 42
    torch.manual_seed(seed)

    model = model.to(device)
    return model

def assess_quality(image_np, model):
    #imt = torch.from_numpy(image_np).float() # This does not work, because dimensions are in different order (probably would work if I reorder them)
    im = Image.fromarray(image_np)
    im = im.convert('RGB')
    
    test_transform = transforms.Compose([
        #transforms.Resize(256),
        #transforms.RandomCrop(224),
        transforms.Resize((224, 224)),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
        ])
    #print("Nothing random going on here")
    imt = test_transform(im)
    imt = imt.unsqueeze(dim=0)
    #print("Tensor size: ", imt.size()) 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    imt = imt.to(device)
    
    model.eval()
    
    with torch.no_grad():
        out = model(imt)
    out = out.view(10, 1)
    # This outputs 10 numbers (summing to 1)
    
    # Let's calculate the mean and std of the distribution
    # This was an incorrect way to calculate std, as it used the wrong mean!
    #mean, std = 0.0, 0.0
    #for j, e in enumerate(out, 1):
    #    mean += j * e
    #    std += e * (j - mean) ** 2
    #std = std ** 0.5
    #print(mean, std)

    predicted_mean, predicted_std = 0.0, 0.0
    for i, elem in enumerate(out, 1):
        predicted_mean += i * elem
    for j, elem in enumerate(out, 1):
        predicted_std += elem * (j - predicted_mean) ** 2
    predicted_std = predicted_std ** 0.5
    #print(predicted_mean, predicted_std)

    # Convert to numpy
    if torch.cuda.is_available():
        predicted_mean = predicted_mean.cpu()
        predicted_std = predicted_std.cpu()
    predicted_mean = predicted_mean.numpy()
    predicted_std = predicted_std.numpy()
    
    return predicted_mean[0], predicted_std[0]
