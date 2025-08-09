def determine_input_size(dataset_name: str, network_name: None) -> int:
    if dataset_name == "MNIST":
        return 28 * 28 if network_name == "simple_mlp_snn" else 1
    elif dataset_name == "CIFAR10":
        return 32 * 32 * 3 if network_name == "simple_mlp_snn" else 3
    else:
        raise ValueError(
            f"Dataset {dataset_name} not recognized for input size determination."
        )
