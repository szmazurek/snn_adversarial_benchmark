def determine_input_size(dataset_name: str, network_name: str) -> int:
    if dataset_name == "MNIST":
        return 28 * 28 if "mlp_snn" in network_name else 1
    elif dataset_name == "CIFAR10":
        return 32 * 32 * 3 if "mlp_snn" in network_name else 3
    elif dataset_name == "EventMNIST":
        return 34 * 34 * 2 if "mlp_snn" in network_name else 2
    else:
        raise ValueError(
            f"Dataset {dataset_name} not recognized for input size determination."
        )
