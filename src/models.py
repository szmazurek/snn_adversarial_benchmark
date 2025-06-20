from torch import nn
from spikingjelly.activation_based import surrogate, neuron, layer
from spikingjelly.activation_based.model.sew_resnet import sew_resnet18


def SewResnet18(
    n_channels: int = 1,
    output_size: int = 10,
    neuron_model: neuron.BaseNode = neuron.LIFNode,
    surrogate_function: surrogate.SurrogateFunctionBase = surrogate.Sigmoid,
) -> nn.Module:
    net = sew_resnet18(
        pretrained=False,
        spiking_neuron=neuron_model,
        cnf="IAND",
        surrogate_function=surrogate_function(),
    )
    net.conv1 = layer.Conv2d(
        n_channels,
        64,
        kernel_size=(7, 7),
        stride=(2, 2),
        padding=(3, 3),
        bias=False,
    )
    net.fc = layer.Linear(512, output_size)
    return net
