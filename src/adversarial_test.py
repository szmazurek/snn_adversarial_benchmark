import os
import json
import torch
import torch.nn as nn
import numpy as np
from os.path import join as path_join
from models import SewResnet18
from datasets import MNISTRepeated
from random import sample as random_sample
from spikingjelly.activation_based import functional, neuron
from torch.nn.functional import softmax

from typing import List, Dict

REPEATS = 10


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
    pred = logit.argmax(dim=1).item()
    return pred == target


def generate_random_frame(img):

    random_img = torch.randn_like(img[0], dtype=torch.float32)
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
    for n, (img, target) in enumerate(dataset):
        img = img.unsqueeze(1)
        pred = model(img).mean(dim=0)
        pred_correct = is_pred_correct(pred, target)

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
        pred_total = torch.zeros_like(pred)
        for f, frame in enumerate(img):
            frame = frame.unsqueeze(0)
            pred = model(frame).mean(dim=0)
            pred_total += pred
            if is_pred_correct(pred, target):
                print(
                    f"Correct prediction achieved at frame {f+1} "
                    f"out of {REPEATS} for sample {n}"
                )
                num_frames_to_solution: int = f + 1
                break
        sample_data_store[n] = {
            "target": target,
            "clean_sample_pred_distribution": softmax(pred, dim=1)
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
                i for i in range(REPEATS) if i not in replaced_frames_idxes
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
                save_path=results_path,
                sample_idx=n,
                label=target,
                is_correct=pred_correct,
                noise_level=replaced_frames_count / REPEATS,
            )
            clear_hook_container(hooked_layers)
            if not pred_correct:
                attack_end = True
        break
    json_results_path = path_join(
        results_path, "adversarial_test_results.json"
    )
    with open(json_results_path, "w") as f:
        json.dump(sample_data_store, f, indent=4)


if __name__ == "__main__":
    model: nn.Module = SewResnet18(n_channels=1)
    functional.set_step_mode(model, step_mode="m")
    model.load_state_dict(torch.load("checkpoints/best_model.pth"))
    mnist_test_set = MNISTRepeated(
        root="./data", train=False, repeat=REPEATS, download=True
    )
    results_path = "./results/"
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    adversarial_attack_test(model, mnist_test_set, results_path)
