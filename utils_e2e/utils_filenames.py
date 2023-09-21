import os

from pathlib import Path


def create_filenames_e2e(buckets, dp_mode, utts_counts_max, args):
    """Create the dictionaries of the classifier checkpoints' filenames."""

    checkpoint_dir = args.checkpoint_dir
    checkpoint_dir_path = Path(checkpoint_dir)
    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)

    filename = f"ckpt_{dp_mode}_dvec_cls_bkts_{buckets}_max_utts_count_{utts_counts_max}_spkperbkt_{args.spk_per_bucket}_agnt_{args.agnt_num}.pt"
    filename_adv = f"ckpt_{dp_mode}_adv_dvec_cls_bkts_{buckets}_max_utts_count_{utts_counts_max}_spkperbkt_{args.spk_per_bucket}_agnt_{args.agnt_num}.pt"

    filename_dir = f"{checkpoint_dir_path}/ckpt_{dp_mode}_dvec_cls_bkts_{buckets}_max_utts_count_{utts_counts_max}_spkperbkt_{args.spk_per_bucket}_agnt_{args.agnt_num}.pt"
    filename_adv_dir = f"{checkpoint_dir_path}/ckpt_{dp_mode}_adv_dvec_cls_bkts_{buckets}_max_utts_count_{utts_counts_max}_spkperbkt_{args.spk_per_bucket}_agnt_{args.agnt_num}.pt"

    filenames_and_dirs = {
        "filename": filename,
        "filename_adv": filename_adv,
        "filename_dir": filename_dir,
        "filename_adv_dir": filename_adv_dir,
    }

    return filenames_and_dirs


def create_filenames_e2e_shadow(buckets, utts_counts_max, args):
    """Create the dictionaries of the classifier checkpoints' filenames."""

    checkpoint_dir = args.checkpoint_dir
    checkpoint_dir_path = Path(checkpoint_dir)
    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)

    filename_shadow = f"ckpt_shadow_dvec_cls_bkts_{buckets}_max_utts_count_{utts_counts_max}_spkperbkt_{args.spk_per_bucket}_agnt_{args.agnt_num}.pt"
    filename_shadow_dir = f"{checkpoint_dir_path}/ckpt_shadow_dvec_cls_bkts_{buckets}_max_utts_count_{utts_counts_max}_spkperbkt_{args.spk_per_bucket}_agnt_{args.agnt_num}.pt"

    filenames_and_dirs = {
        "filename": filename_shadow,
        "filename_dir": filename_shadow_dir,
    }

    return filenames_and_dirs


def create_filenames_memb_inf_e2e(buckets, mode, args):
    """Create the dictionaries of the classifier checkpoints' filenames."""

    checkpoint_dir = args.checkpoint_dir
    checkpoint_dir_path = Path(checkpoint_dir)
    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)

    filename = f"ckpt_memb_inf_{mode}_e2e_bkts_{buckets}_fifty_pcnt_spkperbkt_{args.spk_per_bucket}_agnt_{args.agnt_num}.pt"
    filename_dir = f"{checkpoint_dir_path}/ckpt_memb_inf_{mode}_e2e_bkts_{buckets}_fifty_pcnt_spkperbkt_{args.spk_per_bucket}_agnt_{args.agnt_num}.pt"

    filenames_and_dirs = {
        "filename": filename,
        "filename_dir": filename_dir,
    }

    return filenames_and_dirs


def create_filenames_dp_e2e(buckets, dp_mode, utts_counts_max, eps, delta, args):
    """Create the dictionaries of the classifier checkpoints' filenames."""

    checkpoint_dir = args.checkpoint_dir
    checkpoint_dir_path = Path(checkpoint_dir)
    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)

    filename = f"ckpt_dp_{dp_mode}_eps_{eps:.2f}_delta_{delta}_dvec_cls_bkts_{buckets}_max_utts_count_{utts_counts_max}_spkperbkt_{args.spk_per_bucket}_agnt_{args.agnt_num}.pt"
    filename_dir = f"{checkpoint_dir_path}/ckpt_dp_{dp_mode}_eps_{eps:.2f}_delta_{delta}_dvec_cls_bkts_{buckets}_max_utts_count_{utts_counts_max}_spkperbkt_{args.spk_per_bucket}_agnt_{args.agnt_num}.pt"

    filenames_and_dirs = {
        "filename": filename,
        "filename_dir": filename_dir,
    }

    return filenames_and_dirs


def create_checkpoint_dir(args, filename, filename_dir):
    """Create the available checkpoint for classifier during training."""

    mode = filename.split("_")[1]

    checkpoint_dir = args.checkpoint_dir
    checkpoint_dir_path = Path(checkpoint_dir)
    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)

    file_dir_storage, status_storage = [], []
    for _, _, files in os.walk(checkpoint_dir_path):
        for file in files:
            if file == filename:
                file_dir_storage.append(filename_dir)
                status_storage.append(mode)
            else:
                file_dir_storage.append(None)
                status_storage.append("")

    if filename_dir in file_dir_storage:
        print(f"{mode} checkpoint found")
        file_dir = filename_dir
        status = mode
    else:
        print(f"No checkpoint found")
        file_dir = None
        status = ""

    return file_dir, status


def create_dp_checkpoint_dir(
    args,
    filename,
    filename_dp,
    filename_dir,
    filename_dir_dp,
):
    """Create the available checkpoint for dp classifier during training."""

    mode = filename.split("_")[1]
    mode_dp = filename_dp.split("_")[1]

    checkpoint_dir = args.checkpoint_dir
    checkpoint_dir_path = Path(checkpoint_dir)
    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)

    file_dir_storage, status_storage = [], []
    for _, _, files in os.walk(checkpoint_dir_path):
        for file in files:
            if file == filename_dp:
                file_dir_storage.append(filename_dir_dp)
                status_storage.append(mode_dp)
            elif file == filename:
                file_dir_storage.append(filename_dir)
                status_storage.append(mode)
            else:
                file_dir_storage.append(None)
                status_storage.append("")

    if filename_dir_dp in file_dir_storage:
        print(f"{mode_dp} checkpoint found")
        file_dir = filename_dir_dp
        status = mode_dp
    elif filename_dir in file_dir_storage:
        print(f"{mode} checkpoint found")
        file_dir = filename_dir
        status = mode
    else:
        print(f"No checkpoint found")
        file_dir = None
        status = ""

    return file_dir, status


def create_filenames_results(args, spk_per_bkt, agnt_num):
    """Create filenames for the metrics to be saved as .json files
    together with the corresponding directory paths."""

    # Create paths for saving the training/validation metrics
    result_dir_td = args.result_dir_td
    result_dir_td_path = Path(result_dir_td)
    result_dir_td_path.mkdir(parents=True, exist_ok=True)

    result_dir_acc_val = args.result_dir_acc_val
    result_dir_acc_val_path = Path(result_dir_acc_val)
    result_dir_acc_val_path.mkdir(parents=True, exist_ok=True)

    # Create filenames of the metrics to be saved as .json files
    filename_time_delay = f"td_spkperbkt_{spk_per_bkt}_agnt_{agnt_num}.json"
    filename_acc_val = f"eval_acc_spkperbkt_{spk_per_bkt}_agnt_{agnt_num}.json"

    return {
        "dir_td": result_dir_td_path,
        "dir_acc_val": result_dir_acc_val_path,
        "filename_time_delay": filename_time_delay,
        "filename_acc_val": filename_acc_val,
    }


def create_filenames_shadow_results(args, spk_per_bkt, agnt_num):
    """Create filenames for the metrics to be saved as .json files
    together with the corresponding directory paths."""

    # Create paths for saving the training/validation metrics
    result_dir_td = args.result_dir_td
    result_dir_td_path = Path(result_dir_td)
    result_dir_td_path.mkdir(parents=True, exist_ok=True)

    result_dir_acc_val = args.result_dir_acc_val
    result_dir_acc_val_path = Path(result_dir_acc_val)
    result_dir_acc_val_path.mkdir(parents=True, exist_ok=True)

    # Create filenames of the metrics to be saved as .json files
    filename_time_delay = f"td_shadow_spkperbkt_{spk_per_bkt}_agnt_{agnt_num}.json"
    filename_shadow_acc_val = (
        f"eval_shadow_acc_spkperbkt_{spk_per_bkt}_agnt_{agnt_num}.json"
    )

    return {
        "dir_td": result_dir_td_path,
        "dir_acc_val": result_dir_acc_val_path,
        "filename_time_delay": filename_time_delay,
        "filename_acc_val": filename_shadow_acc_val,
    }


def create_filenames_results_adv_dp(
    args,
    dp_mode,
    buckets,
    utts_counts_max,
    eps,
    delta,
    agnt_num,
):
    """Create filenames for the metrics to be saved as .json files
    together with the corresponding directory paths."""

    # Create paths for saving the training/validation metrics
    result_dir_acc_val_adv_dp = args.result_dir_acc_val_adv_dp
    result_dir_acc_val_adv_dp_path = Path(result_dir_acc_val_adv_dp)
    result_dir_acc_val_adv_dp_path.mkdir(parents=True, exist_ok=True)

    result_dir_privacy_spent = args.result_dir_privacy_spent
    result_dir_privacy_spent_path = Path(result_dir_privacy_spent)
    result_dir_privacy_spent_path.mkdir(parents=True, exist_ok=True)

    # Create filenames of the metrics to be saved as .json files
    filename_acc_val_adv_dp = f"eval_acc_dp_{dp_mode}_eps_{eps:.2f}_delta_{delta}_max_utts_count_{utts_counts_max}_bkt_{buckets}_agnt_{agnt_num}.json"
    filename_privacy_spent = f"privacy_spent_dp_{dp_mode}_eps_{eps:.2f}_delta_{delta}_max_utts_count_{utts_counts_max}_bkt_{buckets}_agnt_{agnt_num}.json"

    return {
        "dir_acc_val_adv_dp": result_dir_acc_val_adv_dp_path,
        "dir_privacy_spent": result_dir_privacy_spent_path,
        "filename_acc_val_adv_dp": filename_acc_val_adv_dp,
        "filename_privacy_spent": filename_privacy_spent,
    }
