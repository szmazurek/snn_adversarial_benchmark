from torch import nn
from spikingjelly.activation_based import surrogate, neuron, layer, encoding
from spikingjelly.activation_based.model.sew_resnet import sew_resnet18
from spikingjelly.activation_based.model.spiking_vgg import spiking_vgg11_bn
from spikingjelly.activation_based.layer import LinearRecurrentContainer
from typing import Dict, Callable, Any


def SewResnet18(
    n_channels: int = 1,
    output_size: int = 10,
    neuron_model: neuron.BaseNode = neuron.LIFNode,
    surrogate_function: surrogate.SurrogateFunctionBase = surrogate.ATan,
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


def SpikingVGG11BN(
    n_channels: int = 1,
    output_size: int = 10,
    neuron_model: neuron.BaseNode = neuron.LIFNode,
    surrogate_function: surrogate.SurrogateFunctionBase = surrogate.ATan,
    remove_last_pool: int = 2,
) -> nn.Module:
    net = spiking_vgg11_bn(
        pretrained=False,
        spiking_neuron=neuron_model,
        surrogate_function=surrogate_function(),
    )

    net.features[0] = layer.Conv2d(
        n_channels,
        64,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        bias=False,
    )
    net.classifier[6] = layer.Linear(4096, output_size)

    pool_indices: list[int] = [
        i
        for i, module in enumerate(net.features)
        if isinstance(module, nn.MaxPool2d)
    ]

    if remove_last_pool > 0 and len(pool_indices) >= remove_last_pool:
        modules_to_keep = []
        last_pool_index_to_keep = (
            pool_indices[len(pool_indices) - remove_last_pool]
            if remove_last_pool < len(pool_indices)
            else -1
        )
        for i, module in enumerate(net.features):
            if i <= last_pool_index_to_keep or not isinstance(
                module, nn.MaxPool2d
            ):
                modules_to_keep.append(module)
        net.features = nn.Sequential(*modules_to_keep)

    return net


def SimpleConvSNN(
    n_channels: int = 1,
    output_size: int = 10,
    neuron_model: neuron.BaseNode = neuron.LIFNode,
    surrogate_function: surrogate.SurrogateFunctionBase = surrogate.ATan,
    native_dvs_input: bool = False,
) -> nn.Module:
    net = nn.Sequential(
        layer.Conv2d(n_channels, 64, kernel_size=3, padding=1, bias=False),
        layer.BatchNorm2d(64),
        neuron_model(surrogate_function=surrogate_function()),
        layer.MaxPool2d(kernel_size=2, stride=2),
        layer.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
        layer.BatchNorm2d(128),
        neuron_model(surrogate_function=surrogate_function()),
        layer.MaxPool2d(kernel_size=2, stride=2),
        layer.AdaptiveAvgPool2d((1, 1)),
        layer.Flatten(),
        layer.Linear(128, 64),
        layer.BatchNorm1d(64),
        neuron_model(surrogate_function=surrogate_function()),
        layer.Linear(64, output_size),
    )
    if not native_dvs_input:
        encoder = encoding.PoissonEncoder()
        net = nn.Sequential(encoder, net)

    return net


def SimpleConvSNNRecurrent(
    n_channels: int = 1,
    output_size: int = 10,
    neuron_model: neuron.BaseNode = neuron.LIFNode,
    surrogate_function: surrogate.SurrogateFunctionBase = surrogate.ATan,
    native_dvs_input: bool = False,
) -> nn.Module:
    net = nn.Sequential(
        layer.Conv2d(n_channels, 64, kernel_size=3, padding=1, bias=False),
        layer.BatchNorm2d(64),
        neuron_model(surrogate_function=surrogate_function()),
        layer.MaxPool2d(kernel_size=2, stride=2),
        layer.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
        layer.BatchNorm2d(128),
        neuron_model(surrogate_function=surrogate_function()),
        layer.MaxPool2d(kernel_size=2, stride=2),
        layer.AdaptiveAvgPool2d((1, 1)),
        layer.Flatten(),
        layer.Linear(128, 64),
        layer.BatchNorm1d(64),
        LinearRecurrentContainer(
            neuron_model(surrogate_function=surrogate_function()),
            in_features=64,
            out_features=64,
        ),
        layer.Linear(64, output_size),
    )
    if not native_dvs_input:
        encoder = encoding.PoissonEncoder()
        net = nn.Sequential(encoder, net)
    return net


def _simple_MLP_SNN(
    n_channels: int = 784,
    output_size: int = 10,
    neuron_model: neuron.BaseNode = neuron.LIFNode,
    surrogate_function: surrogate.SurrogateFunctionBase = surrogate.ATan,
    native_dvs_input: bool = False,
) -> nn.Module:
    net = nn.Sequential(
        layer.Linear(n_channels, 512),
        layer.BatchNorm1d(512),
        neuron_model(surrogate_function=surrogate_function()),
        layer.Linear(512, 256),
        layer.BatchNorm1d(256),
        neuron_model(surrogate_function=surrogate_function()),
        layer.Linear(256, 64),
        layer.BatchNorm1d(64),
        neuron_model(surrogate_function=surrogate_function()),
        layer.Linear(64, output_size),
    )
    if not native_dvs_input:
        encoder = encoding.PoissonEncoder()
        flatten = layer.Flatten()
        net = nn.Sequential(encoder, flatten, net)
    return net


def _simple_MLP_SNN_recurrent(
    n_channels: int = 784,
    output_size: int = 10,
    neuron_model: neuron.BaseNode = neuron.LIFNode,
    surrogate_function: surrogate.SurrogateFunctionBase = surrogate.ATan,
    native_dvs_input: bool = False,
) -> nn.Module:
    net = nn.Sequential(
        layer.Linear(n_channels, 512),
        layer.BatchNorm1d(512),
        LinearRecurrentContainer(
            neuron_model(surrogate_function=surrogate_function()),
            in_features=512,
            out_features=512,
        ),
        layer.Linear(512, 256),
        layer.BatchNorm1d(256),
        LinearRecurrentContainer(
            neuron_model(surrogate_function=surrogate_function()),
            in_features=256,
            out_features=256,
        ),
        layer.Linear(256, 64),
        layer.BatchNorm1d(64),
        LinearRecurrentContainer(
            neuron_model(surrogate_function=surrogate_function()),
            in_features=64,
            out_features=64,
        ),
        layer.Linear(64, output_size),
    )
    if not native_dvs_input:
        encoder = encoding.PoissonEncoder()
        flatten = layer.Flatten()
        net = nn.Sequential(encoder, flatten, net)
    return net


class SimpleMLPSNNRecurrent(nn.Module):
    def __init__(
        self,
        n_channels: int = 784,
        output_size: int = 10,
        neuron_model: neuron.BaseNode = neuron.LIFNode,
        surrogate_function: surrogate.SurrogateFunctionBase = surrogate.ATan,
        native_dvs_input: bool = False,
    ):
        super(SimpleMLPSNNRecurrent, self).__init__()
        self.model = _simple_MLP_SNN_recurrent(
            n_channels=n_channels,
            output_size=output_size,
            neuron_model=neuron_model,
            surrogate_function=surrogate_function,
        )
        self.encoder = encoding.PoissonEncoder()
        self.flatten = layer.Flatten()
        self.native_dvs_input = native_dvs_input

    def forward(self, x):

        if not self.native_dvs_input:
            poisson_spikes = self.encoder(x)
            x = self.flatten(poisson_spikes)
        return self.model(x)


class SimpleMLPSNN(nn.Module):
    def __init__(
        self,
        n_channels: int = 784,
        output_size: int = 10,
        neuron_model: neuron.BaseNode = neuron.LIFNode,
        surrogate_function: surrogate.SurrogateFunctionBase = surrogate.ATan,
        native_dvs_input: bool = False,
    ):
        super(SimpleMLPSNN, self).__init__()
        self.model = _simple_MLP_SNN(
            n_channels=n_channels,
            output_size=output_size,
            neuron_model=neuron_model,
            surrogate_function=surrogate_function,
        )
        self.encoder = encoding.PoissonEncoder()
        self.flatten = layer.Flatten()
        self.native_dvs_input = native_dvs_input

    def forward(self, x):
        if not self.native_dvs_input:
            poisson_spikes = self.encoder(x)
            x = self.flatten(poisson_spikes)
        return self.model(x)


MODEL_MAP: Dict[str, Callable[[Any], nn.Module]] = {
    "sew_resnet": SewResnet18,
    "spiking_vgg": SpikingVGG11BN,
    "simple_conv_snn": SimpleConvSNN,
    "simple_mlp_snn": SimpleMLPSNN,
    "simple_conv_snn_recurrent": SimpleConvSNNRecurrent,
    "simple_mlp_snn_recurrent": SimpleMLPSNNRecurrent,
}
