import json

from pathlib import Path


def create_dataset_arguments(args, dir):
    """Create dataset arguments.

    Args:
        - args: necessary arguments to create the dataset.
        - dir: base directory name.

    Returns:
        - data_dir (Union[str, Path]): data directory.
        - speaker_infos (dict): speaker information.

    """

    data_dir = f"{dir}_{args.agnt_num}/" if args.agnt_num > 0 else dir

    with open(Path(data_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)
    speaker_infos = metadata["speaker_gender"]

    return data_dir, speaker_infos


def create_dataset_arguments_bkt(args, bkt, dir_base_name, utts_counts_max):
    """Create dataset arguments.

    Args:
        - args: necessary arguments to create the dataset.
        - bkt: composite bucket.
        - utts_counts_max: max utts counter.
        - dir_base_name: base directory name.

    Returns:
        - data_dir (Union[str, Path]): data directory.
        - speaker_infos (dict): speaker information.

    """

    data_dir = f"{dir_base_name}_bkt_{bkt}_max_utts_count_{utts_counts_max}_agnt_{args.agnt_num}/"

    with open(Path(data_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)
    speaker_infos = metadata["speaker_gender"]

    return data_dir, speaker_infos


def create_dataset_arguments_validation(args, dir):
    """Create dataset arguments for validation.

    Args:
        - args: necessary arguments to create the dataset.
        - dir: base directory name.

    Returns:
        - validation_data_dir (Union[str, Path]): validation data directory.
        - speaker_infos_validation (dict): speaker information.

    """

    validation_data_dir = f"{dir}_{args.agnt_num}/" if args.agnt_num > 0 else dir

    with open(Path(validation_data_dir, "metadata.json"), "r") as f_validation:
        metadata_validation = json.load(f_validation)
    speaker_infos_validation = metadata_validation["speaker_gender"]

    return validation_data_dir, speaker_infos_validation
