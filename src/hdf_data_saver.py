import h5py
import torch

from typing import Dict, List, Optional


class HDF5Manager:
    """
    Manages saving data to an HDF5 file, mirroring a directory structure.

    This class provides an interface to incrementally save data, such as simulation
    results, into an HDF5 file, organizing it in a hierarchical structure based on
    labels, sample indices, and conditions (correct/incorrect, noise levels).

    Attributes:
        filepath (str): The path to the HDF5 file.
        mode (str): The file access mode ('w' for write, 'a' for append).  Defaults to 'a'.
        _file (h5py.File): The HDF5 file object (opened internally).

    Example Usage:
        manager = HDF5Manager("simulation_results.hdf5")
        for sample_idx, (label, is_correct, noise_level, data) in enumerate(simulation_data):
            manager.save_data(label, sample_idx, is_correct, noise_level, data)
        manager.close()  # Remember to close the file when finished.

    """

    def __init__(self, filepath: str, mode: str = "a"):
        """
        Initializes the HDF5Manager.

        Args:
            filepath (str): Path to the HDF5 file.
            mode (str, optional): File access mode ('w' for write, 'a' for append).
                Defaults to 'a'.  Use 'w' to create a new file or overwrite an existing one.
                Use 'a' to append to an existing file.
        """
        self.filepath = filepath
        self.mode = mode
        self._file = None  # HDF5 file object, opened on demand

    def _open_file(self):
        """Opens the HDF5 file.  Internal method."""
        if self._file is None:
            self._file = h5py.File(self.filepath, self.mode)

    def _close_file(self):
        """Closes the HDF5 file.  It's the user's responsibility to call this."""
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self):
        """Supports using the class in a 'with' statement (context manager)."""
        self._open_file()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Closes the file automatically when exiting the 'with' block."""
        self._close_file()

    def save_data(
        self,
        label: int,
        sample_idx: int,
        is_correct: bool,
        noise_level: Optional[float],
        hooked_layers: Dict[str, Dict[str, List[torch.Tensor]]],
    ):
        """
        Saves data for a single sample into the HDF5 file.

        Args:
            label (int): The label of the sample.
            sample_idx (int): The index of the sample.
            is_correct (bool): Whether the sample was correctly classified.
            noise_level (Optional[float]): The noise level, if applicable.  If None,
                it's assumed to be the original data.
            hooked_layers (Dict[str, Dict[str, List[torch.Tensor]]]):  The data
                to save, typically captured from network hooks.  The structure
                is assumed to be:
                {
                    layer_name: {
                        'v_seq': list of torch.Tensor,
                        's_seq': list of torch.Tensor
                    },
                    ...
                }
        """
        self._open_file()  # Ensure file is open

        # Construct the group path, mirroring the directory structure
        group_path = f"label_{label}/sample_{sample_idx}/"
        last_dir_name = "correct" if is_correct else "incorrect"
        last_dir_name += (
            "_original"
            if noise_level is None
            else f"_noise_{int(100 * noise_level)}"
        )
        group_path += last_dir_name

        # Create the group if it doesn't exist.  This ensures the hierarchy.
        if group_path not in self._file:
            self._file.create_group(group_path)

        # Save the data within the group
        for layer_name, data in hooked_layers.items():
            # Convert the list of tensors to a single numpy array
            v_seq = torch.cat(data["v_seq"]).cpu().numpy().squeeze()
            s_seq = torch.cat(data["s_seq"]).cpu().numpy().squeeze()
            # Replace '-' with 'm' in layer name for HDF5 compatibility
            safe_layer_name = layer_name.replace("-", "m")

            # Use create_dataset, and check for existence to avoid overwriting
            v_dataset_name = f"{safe_layer_name}_voltage"
            s_dataset_name = f"{safe_layer_name}_spike"

            if v_dataset_name in self._file[group_path]:
                del self._file[group_path][v_dataset_name]  # Delete if exists
            if s_dataset_name in self._file[group_path]:
                del self._file[group_path][s_dataset_name]  # Delete if exists

            self._file.create_dataset(
                f"{group_path}/{v_dataset_name}", data=v_seq
            )
            self._file.create_dataset(
                f"{group_path}/{s_dataset_name}", data=s_seq
            )

        self._file.flush()  # Ensure data is written to the file

    def close(self):
        """Closes the HDF5 file.  Important to call this when finished."""
        self._close_file()

    # The following methods are less relevant to the user's immediate request,
    # but are included for completeness and potential future use.  They are
    # NOT used in the provided solution.

    def load_data(self, group_path: str, layer_name: str):
        """
        Loads data for a specific layer from a group.

        Args:
            group_path (str): The path to the group containing the data.
            layer_name (str): The name of the layer (used to construct the
                dataset names).

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]: The voltage and spike data
                for the specified layer, or (None, None) if the data is not found.
        """
        self._open_file()  # Ensure file is open

        safe_layer_name = layer_name.replace("-", "m")
        v_dataset_name = f"{safe_layer_name}_voltage"
        s_dataset_name = f"{safe_layer_name}_spike"

        if group_path in self._file:
            group = self._file[group_path]
            if v_dataset_name in group and s_dataset_name in group:
                v_seq = group[v_dataset_name][()]  # Read the data
                s_seq = group[s_dataset_name][()]
                return v_seq, s_seq
            else:
                return None, None
        else:
            return None, None

    def get_all_group_names(self):
        """
        Retrieves a list of all group names in the HDF5 file.  Useful for
        inspecting the file structure.

        Returns:
            list: A list of strings, where each string is the full path to a group.
        """
        self._open_file()  # ensure file is open

        def _visitor_func(name, obj):
            if isinstance(obj, h5py.Group):
                groups.append(name)

        groups = []
        self._file.visititems(_visitor_func)
        return groups

    def delete_group(self, group_path: str):
        """
        Deletes a group and all its contents from the HDF5 file.  Use with caution!

        Args:
            group_path (str): The path to the group to delete.
        """
        self._open_file()  # ensure file is open
        if group_path in self._file:
            del self._file[group_path]
            self._file.flush()  # Ensure deletion is persisted
        else:
            print(f"Warning: Group '{group_path}' not found, cannot delete.")
