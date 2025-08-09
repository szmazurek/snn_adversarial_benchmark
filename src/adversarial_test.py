import os
import json
import torch
import torch.nn as nn
import numpy as np
from os.path import join as path_join

import tqdm
from datasets import DatasetFactory
from random import sample as random_sample
from spikingjelly.activation_based import functional, neuron
from torch.nn.functional import softmax

from typing import List, Dict
from datasets import DatasetFactory
from models import MODEL_MAP
from utils import determine_input_size
from argparse import ArgumentParser


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_lif_nodes(model: nn.Module) -> List[nn.Module]:

    nodes_indices: List[nn.Module] = []
    for _, layer_object in model.named_modules():
        if isinstance(layer_object, neuron.LIFNode):
            nodes_indices.append(layer_object)
    return nodes_indices


def apply_forward_hooks(
    lif_module_list: List[nn.Module],
) -> Dict[str, Dict[str, List[torch.Tensor]]]:
    def make_hook(layer_key):
        def hook(m, x, y):
            hooked_layers[layer_key]["v_seq"].append(m.v.unsqueeze(0))
            hooked_layers[layer_key]["s_seq"].append(y.unsqueeze(0))

        return hook

    hooked_layers = {}
    for idx, lif_node in enumerate(lif_module_list):
        layer_name = f"layer_{idx}"
        hooked_layers[layer_name] = {
            "layer": lif_node,
            "v_seq": [],
            "s_seq": [],
        }
        lif_node.register_forward_hook(make_hook(layer_name))
    return hooked_layers


def clear_hook_container(
    hooked_layers: Dict[str, Dict[str, List[torch.Tensor]]],
) -> Dict[str, Dict[str, List[torch.Tensor]]]:
    for name, _ in hooked_layers.items():
        hooked_layers[name]["v_seq"] = []
        hooked_layers[name]["s_seq"] = []


def process_data_recorded_by_hooks(
    hooked_layers: Dict[str, Dict[str, List[torch.Tensor]]],
    save_path: str,
    sample_idx: int,
    label: int,
    is_correct: bool,
    noise_level: float | None = None,
):
    sample_save_path: str = path_join(
        save_path, f"label_{label}", f"sample_{sample_idx}"
    )
    last_dir_name: str = "correct" if is_correct else "incorrect"
    last_dir_name += (
        "_original"
        if noise_level is None
        else f"_noise_{round(100*noise_level)}"
    )
    sample_save_path = path_join(sample_save_path, last_dir_name)
    if not os.path.exists(sample_save_path):
        os.makedirs(sample_save_path)
    for name, data in hooked_layers.items():
        v_seq = torch.cat(data["v_seq"]).cpu().numpy().squeeze()
        s_seq = torch.cat(data["s_seq"]).cpu().numpy().squeeze()
        voltage_path = path_join(
            sample_save_path,
            f"test_{sample_idx}_label_{label}_voltage"
            f"_{name.replace('-', 'm')}.npy",
        )
        spike_path = path_join(
            sample_save_path,
            f"test_{sample_idx}_label_{label}_spike_"
            f"{name.replace('-', 'm')}.npy",
        )
        np.save(voltage_path, v_seq)
        np.save(spike_path, s_seq)


def is_pred_correct(logit, target):
    pred = logit.argmax(dim=1).cpu().item()
    return pred == target


def generate_random_frame(img):

    random_img = torch.randn_like(img[0], dtype=torch.float32, device=DEVICE)
    # TODO should I standardize with the dataset mean and std?
    random_img = (random_img - random_img.min()) / (
        random_img.max() - random_img.min()
    )
    return random_img


@torch.no_grad()
def adversarial_attack_test(
    args,
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
):
    model.eval()
    lif_nodes: List[nn.Module] = get_lif_nodes(model)
    hooked_layers: Dict[str, Dict[str, List[torch.Tensor]]] = (
        apply_forward_hooks(lif_nodes)
    )
    sample_data_store = {}
    progbar = tqdm.tqdm(
        dataset,
        total=len(dataset),
        desc="Adversarial attack test",
        unit="sample",
    )
    json_results_path = path_join(
        args.results_dir, "adversarial_test_results.json"
    )
    correct_preds = 0
    for n, (img, target) in enumerate(progbar):
        if correct_preds >= args.n_samples_to_asses:
            break
        img = img.unsqueeze(1).to(DEVICE)
        pred_original = model(img).mean(dim=0)
        pred_correct = is_pred_correct(pred_original, target)

        if not pred_correct:
            continue
        correct_preds += 1
        process_data_recorded_by_hooks(
            hooked_layers=hooked_layers,
            save_path=args.results_dir,
            sample_idx=n,
            label=target,
            is_correct=pred_correct,
            noise_level=None,
        )
        # save metadata of original sample and pred logits (distirbution)
        # estimate time to achieve correct prediction
        functional.reset_net(model)
        pred_total = torch.zeros_like(pred_original, device=DEVICE)
        for f, frame in enumerate(img):
            frame = frame.unsqueeze(0)
            pred = model(frame).mean(dim=0)
            pred_total += pred
            if is_pred_correct(pred_total, target):

                num_frames_to_solution: int = f + 1
                break
        sample_data_store[n] = {
            "target": target,
            "clean_sample_pred_distribution": softmax(pred_original, dim=1)
            .cpu()
            .squeeze()
            .tolist(),
            "num_frames_to_solve": num_frames_to_solution,
            "adversarial_samples_results": [],
        }
        attack_end = False
        replaced_frames_idxes = []
        replaced_frames_count = 0
        adversarial_img = img.clone()
        clear_hook_container(hooked_layers)

        while not attack_end:
            functional.reset_net(model)
            # randomly select a frame to replace
            # (but not the ones already replaced)
            idxs_to_choose = [
                i
                for i in range(args.repeats)
                if i not in replaced_frames_idxes
            ]
            if not idxs_to_choose:
                attack_end = True
                break
            replace_idx = random_sample(idxs_to_choose, 1)[0]
            adversarial_img[replace_idx] = generate_random_frame(img)
            replaced_frames_idxes.append(replace_idx)
            replaced_frames_count += 1
            # predict
            pred = model(adversarial_img).mean(dim=0)
            pred_correct = is_pred_correct(pred, target)
            # save metadata of adversarial sample and pred
            # logits (distirbution)
            sample_data_store[n]["adversarial_samples_results"].append(
                {
                    "pred_distribution": softmax(pred, dim=1)
                    .cpu()
                    .squeeze()
                    .tolist(),
                    "pred_correct": pred_correct,
                    "replaced_frames_count": replaced_frames_count,
                    "replaced_frames_idxes": replaced_frames_idxes,
                    "current_attack_replace_idx": replace_idx,
                }
            )
            process_data_recorded_by_hooks(
                hooked_layers=hooked_layers,
                save_path=args.results_dir,
                sample_idx=n,
                label=target,
                is_correct=pred_correct,
                noise_level=replaced_frames_count / args.repeats,
            )
            clear_hook_container(hooked_layers)
            if not pred_correct:
                attack_end = True

        with open(json_results_path, "w") as f:
            json.dump(sample_data_store, f, indent=4)


if __name__ == "__main__":
    parser = ArgumentParser(description="Adversarial Attack Test Script")

    parser.add_argument(
        "--n_samples_to_asses",
        type=int,
        default=2000,
        help="Number of samples to assess in the dataset",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="Number of repeated frames in the input data",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="default_experiment",
        help="Name of the experiment, used for checkpointing",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sew_resnet",
        choices=MODEL_MAP.keys(),
        help=f"Model architecture to use. Available options: {', '.join(MODEL_MAP.keys())}",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="MNIST",
        choices=["MNIST", "CIFAR10"],
        help="Dataset to use for training and testing",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Whether to z-score normalize the dataset. If false, min-max scaling is applied.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        help="Directory to save results",
    )

    args = parser.parse_args()
    model = MODEL_MAP[args.model](
        n_channels=determine_input_size(args.dataset, args.model),
    )
    functional.set_step_mode(model, step_mode="m")
    checkpoint_path = path_join(
        args.checkpoint_dir, f"{args.experiment_name}_best.pth"
    )
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint file {checkpoint_path} does not exist."
        )
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.to(DEVICE)
    mnist_test_set = DatasetFactory.create_dataset(
        args.dataset,
        root="./data",
        train=False,
        repeat=args.repeats,
        download=True,
        normalize=args.normalize,
    )
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    adversarial_attack_test(args, model, mnist_test_set)
