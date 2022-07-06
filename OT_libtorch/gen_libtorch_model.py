#!/usr/bin/env python3
# Developed by Junyi Ma, Xieyuanli Chen, and Jun Zhang
# This file is covered by the LICENSE file in the root of the project OverlapTransformer:
# https://github.com/haomo-ai/OverlapTransformer/
# Brief: Generate OT model for Libtorch

import os
import sys
p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from modules.overlap_transformer import featureExtracter
import yaml

# load config ================================================================
config_filename = '../config/config.yml'
config = yaml.safe_load(open(config_filename))
test_weights = config["test_config"]["test_weights"]
# ============================================================================


print(test_weights)
checkpoint = torch.load(test_weights)
# for KITTI
amodel = featureExtracter(height=64, width=900, channels=1, use_transformer=True)
# for Haomo
# amodel = featureExtracter(height=32, width=900, channels=1, use_transformer=True)
amodel.load_state_dict(checkpoint['state_dict']) 

amodel.cuda()
amodel.eval()

# 64 for KITTI
example = torch.rand(1, 1, 64, 900)   
# 32 for Haomo
# example = torch.rand(1, 1, 32, 900)   
example = example.cuda()
# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(amodel, example)
traced_script_module.save("./overlapTransformer.pt")
