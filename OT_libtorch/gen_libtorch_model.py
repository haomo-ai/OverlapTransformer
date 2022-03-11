import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from modules.feature_extracter_without_delta_layer import featureExtracter



checkpoint = torch.load("/home/mjy/dev/OverlapNet++/tools/amodel_transformer_depth_only19.pth.tar")
amodel = featureExtracter(channels=1, use_transformer=True)
amodel.load_state_dict(checkpoint['state_dict']) 

amodel.cuda()
amodel.eval()

example = torch.rand(1, 1, 32, 900)
example = example.cuda()
# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(amodel, example)
traced_script_module.save("./overlapTransformer.pt")


