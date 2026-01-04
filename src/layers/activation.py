import torch.nn as nn

def get_activation(name="relu"):
    if name.lower() == "relu":
        return nn.ReLU(inplace=True)
    elif name.lower() == "sigmoid":
        return nn.Sigmoid()
    else:
        raise ValueError(f"{name} activation not implemented")
