import json
import torch
import torch.nn as nn


from torch.optim import Adam
from utils_e2e import (
    parse_args,
    HyperParamsDP,
    create_filenames_e2e,
    create_filenames_dp_e2e,
    create_filenames_e2e_shadow,
    create_filenames_memb_inf_e2e,
    create_checkpoint_dir,
    create_dp_checkpoint_dir,
    MembInfAttk_v3,
    DvecAttentivePooledClsE2E_v3,
    ShadowDvecAttentivePooledClsE2E,
)


from training_functions_e2e import train_memb_inf_attk_relativistic
from attack_strategy_e2e import CWSpkIDE2E, NonTargetedCostCW, NonTargetLabelStrategy


def main():
    args = parse_args()
    hparams = HyperParamsDP()

    # Set to appropriate modes to design different types of attack
    mode = "shadow_model_adv"
    dp_mode = "full"

    # Select the appropriate utts percentage based on max utts counter
    utts_counts_max = 5

    # Optimizer
    opt = {"opt": Adam}

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    filenames_shadow_dvecs_cls_and_dirs = create_filenames_e2e_shadow(
        hparams.buckets,
        utts_counts_max,
        args,
    )
    filenames_dvecs_cls_and_dirs = create_filenames_e2e(
        hparams.buckets,
        dp_mode,
        utts_counts_max,
        args,
    )
    filenames_dvecs_cls_and_dirs_dp = create_filenames_dp_e2e(
        hparams.buckets_dp,
        dp_mode,
        utts_counts_max,
        2.25,
        hparams.delta,
        args,
    )

    filenames_and_dirs_memb_inf = create_filenames_memb_inf_e2e(
        hparams.buckets,
        mode,
        args,
    )

    # Set ``ckpt_cls'' to the available checkpoint
    if mode == "shadow_model":
        ckpt_shadow_cls, status = create_checkpoint_dir(
            args,
            filenames_shadow_dvecs_cls_and_dirs["filename"],
            filenames_shadow_dvecs_cls_and_dirs["filename_dir"],
        )

        # Model
        model_collection = {
            "memb_attk_model": MembInfAttk_v3,
            "shadow_model": ShadowDvecAttentivePooledClsE2E,
        }
    elif mode == "shadow_model_adv":
        ckpt_shadow_cls, status = create_checkpoint_dir(
            args,
            filenames_dvecs_cls_and_dirs["filename_adv"],
            filenames_dvecs_cls_and_dirs["filename_adv_dir"],
        )

        # Model
        model_collection = {
            "memb_attk_model": MembInfAttk_v3,
            "shadow_model": DvecAttentivePooledClsE2E_v3,
        }
    elif mode == "shadow_model_dp":
        ckpt_shadow_cls, status = create_dp_checkpoint_dir(
            args,
            filenames_dvecs_cls_and_dirs["filename_adv"],
            filenames_dvecs_cls_and_dirs_dp["filename"],
            filenames_dvecs_cls_and_dirs["filename_adv_dir"],
            filenames_dvecs_cls_and_dirs_dp["filename_dir"],
        )

        # Model
        model_collection = {
            "memb_attk_model": MembInfAttk_v3,
            "shadow_model": DvecAttentivePooledClsE2E_v3,
        }

    ckpt_memb_inf_cls, _ = create_checkpoint_dir(
        args,
        filenames_and_dirs_memb_inf["filename"],
        filenames_and_dirs_memb_inf["filename_dir"],
    )

    attk_props = {
        "name": CWSpkIDE2E,
        "cost": NonTargetedCostCW(device),
        "tgt_label": NonTargetLabelStrategy(device),
        "attk_type": "CW",
    }

    loss_props = {"loss": nn.BCEWithLogitsLoss}

    train_memb_inf_attk_relativistic(
        args,
        hparams,
        device,
        model_collection,
        opt,
        loss_props,
        attk_props,
        status,
        ckpt_shadow_cls,
        ckpt_memb_inf_cls,
        utts_counts_max,
        filenames_and_dirs_memb_inf,
    )


if __name__ == "__main__":
    main()
