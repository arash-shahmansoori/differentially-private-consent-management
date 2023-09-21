import torch

from torch.optim import Adam
from utils_e2e import (
    create_filenames_e2e,
    parse_args,
    HyperParamsDP,
    create_checkpoint_dir,
    create_filenames_e2e_shadow,
    DvecAttentivePooledClsE2E,
    DvecAttentivePooledClsE2E_v3,
    ShadowDvecAttentivePooledClsE2E,
)


from attack_strategy_e2e import CWSpkIDE2E, NonTargetedCostCW, NonTargetLabelStrategy
from training_functions_e2e import train_adv_comp_e2e


def main():
    args = parse_args()
    hparams = HyperParamsDP()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    # Select the appropriate utts percentage based on max utts counter
    utts_counts_max = 5

    dp_mode = "selective"  # mode for dp training, e.g., full, selective, BiTFiT

    if dp_mode == "full" or dp_mode == "BiTFiT":
        # Model
        model_collection = {
            "model": DvecAttentivePooledClsE2E_v3,
            "shadow_model": ShadowDvecAttentivePooledClsE2E,
        }

    elif dp_mode == "selective":
        # Model
        model_collection = {
            "model": DvecAttentivePooledClsE2E,
            "shadow_model": ShadowDvecAttentivePooledClsE2E,
        }

    # Optimizer
    opt = {"opt": Adam}

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

    # Set ``ckpt_cls_shadow'' and ``ckpt_cls'' to the available checkpoint
    ckpt_dvecs_cls_shadow, _ = create_checkpoint_dir(
        args,
        filenames_shadow_dvecs_cls_and_dirs["filename"],
        filenames_shadow_dvecs_cls_and_dirs["filename_dir"],
    )

    ckpt_dvecs_cls, status_dvecs_cls = create_checkpoint_dir(
        args,
        filenames_dvecs_cls_and_dirs["filename_adv"],
        filenames_dvecs_cls_and_dirs["filename_adv_dir"],
    )

    attk_props = {
        "name": CWSpkIDE2E,
        "cost": NonTargetedCostCW(device),
        "tgt_label": NonTargetLabelStrategy(device),
        "attk_type": "CW",
    }

    train_adv_comp_e2e(
        args,
        hparams,
        device,
        model_collection,
        opt,
        attk_props,
        ckpt_dvecs_cls_shadow,
        ckpt_dvecs_cls,
        status_dvecs_cls,
        utts_counts_max,
        filenames_dvecs_cls_and_dirs,
    )


if __name__ == "__main__":
    main()
