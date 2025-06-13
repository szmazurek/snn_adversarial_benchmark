import os
import json
import torch
import torch.nn as nn
import numpy as np
from os.path import join as path_join
from itertools import combinations
import tqdm
from models import SewResnet18
from datasets import MNISTRepeated
from random import sample as random_sample
from spikingjelly.activation_based import functional, neuron
from torch.nn.functional import softmax

from typing import List, Dict


REPEATS = 10
N_ITERS_NOISE_INJECTION = 10
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


def process_data_recorded_by_hooks_for_avg(
    hooked_layers: Dict[str, Dict[str, List[torch.Tensor]]],
    summed_results_hooked_layers: Dict[str, Dict[str, List[torch.Tensor]]],
):
    for name, data in hooked_layers.items():
        concatenated_v_seq = torch.cat(data["v_seq"])
        concatenated_s_seq = torch.cat(data["s_seq"])
        if name not in summed_results_hooked_layers:
            summed_results_hooked_layers[name] = {
                "v_seq": concatenated_v_seq,
                "s_seq": concatenated_s_seq,
            }
        else:

            summed_results_hooked_layers[name]["v_seq"] += concatenated_v_seq
            summed_results_hooked_layers[name]["s_seq"] += concatenated_s_seq


def process_data_recorded_by_hooks_avg(
    summed_results_hooked_layers: Dict[str, torch.Tensor],
    n_sample_repeats: int,
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
    for name, data in summed_results_hooked_layers.items():
        v_seq = (data["v_seq"] / n_sample_repeats).cpu().numpy().squeeze()
        s_seq = (data["s_seq"] / n_sample_repeats).cpu().numpy().squeeze()
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
    # CAUTION: what happens if we go out of the range of 0-1?
    # Will it crash earlier?
    random_img = (random_img - random_img.min()) / (
        random_img.max() - random_img.min()
    )
    return random_img


@torch.no_grad()
def adversarial_attack_test(
    model: nn.Module, dataset: torch.utils.data.Dataset, results_path: str
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

    for n, (img, target) in enumerate(progbar):
        img = img.unsqueeze(1).to(DEVICE)
        pred_original = model(img).mean(dim=0)
        pred_correct = is_pred_correct(pred_original, target)

        if not pred_correct:
            continue
        process_data_recorded_by_hooks(
            hooked_layers=hooked_layers,
            save_path=results_path,
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
        adversarial_img = img.clone()
        clear_hook_container(hooked_layers)

        for evaluated_noise_level in range(1, REPEATS):
            attack_stop = False
            functional.reset_net(model)
            summed_results_hooked_layers = {}

            idx_to_replace_combinations = list(
                combinations(range(REPEATS), evaluated_noise_level)
            )
            idx_to_replace_combinations = random_sample(
                idx_to_replace_combinations, N_ITERS_NOISE_INJECTION
            )
            preds_correctness = []
            for replace_idxes in idx_to_replace_combinations:

                for replace_idx in replace_idxes:
                    adversarial_img[replace_idx] = generate_random_frame(img)

                pred = model(adversarial_img).mean(dim=0)
                preds_correctness.append(is_pred_correct(pred, target))
                process_data_recorded_by_hooks_for_avg(
                    hooked_layers=hooked_layers,
                    summed_results_hooked_layers=summed_results_hooked_layers,
                )
                clear_hook_container(hooked_layers)
                functional.reset_net(model)
                adversarial_img = img.clone()
            if any(not correctness for correctness in preds_correctness):
                attack_stop = True
            process_data_recorded_by_hooks_avg(
                summed_results_hooked_layers=summed_results_hooked_layers,
                n_sample_repeats=N_ITERS_NOISE_INJECTION,
                save_path=results_path,
                sample_idx=n,
                label=target,
                is_correct=True if not attack_stop else False,
                noise_level=evaluated_noise_level / REPEATS,
            )
            if attack_stop:
                break


if __name__ == "__main__":
    model: nn.Module = SewResnet18(n_channels=1)
    functional.set_step_mode(model, step_mode="m")
    model.load_state_dict(torch.load("../checkpoints/best_model.pth"))
    model.to(DEVICE)
    mnist_test_set = MNISTRepeated(
        root="./data", train=False, repeat=REPEATS, download=True
    )
    results_path = "./results_avg/"
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    adversarial_attack_test(model, mnist_test_set, results_path)
