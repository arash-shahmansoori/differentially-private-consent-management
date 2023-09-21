import torch
import torch.nn as nn

from utils_e2e import (
    parse_args,
    HyperParamsDP,
    create_filenames_e2e,
    create_filenames_dp_e2e,
    create_filenames_memb_inf_e2e,
    create_filenames_e2e_shadow,
    create_checkpoint_dir,
    create_dp_checkpoint_dir,
    DvecAttentivePooledClsE2E_v3,
    ShadowDvecAttentivePooledClsE2E,
    MembInfAttk_v3,
)


from evaluation_e2e import eval_memb_inf_performance_on_dp_model_e2e_relativistic

from attack_strategy_e2e import (
    CWSpkIDE2E,
    NonTargetedCostCW,
    NonTargetLabelStrategy,
)


def main():
    args = parse_args()
    hparams = HyperParamsDP()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    # Set to appropriate modes
    dp_mode = "BiTFiT"

    # Select the appropriate utts percentage based on max utts counter
    utts_counts_max = 5

    filenames_shadow_dvecs_cls_and_dirs = create_filenames_e2e_shadow(
        hparams.buckets,
        utts_counts_max,
        args,
    )

    filenames_dvecs_cls_and_dirs = create_filenames_e2e(
        hparams.buckets,
        "full",
        utts_counts_max,
        args,
    )
    filenames_dvecs_cls_and_dirs_dp = create_filenames_dp_e2e(
        hparams.buckets_dp,
        dp_mode,
        utts_counts_max,
        2.27,
        hparams.delta,
        args,
    )

    filenames_and_dirs_memb_inf = create_filenames_memb_inf_e2e(
        hparams.buckets,
        "shadow_model_adv",
        args,
    )

    ckpt_dvec_cls_shadow, _ = create_checkpoint_dir(
        args,
        filenames_dvecs_cls_and_dirs["filename_adv"],
        filenames_dvecs_cls_and_dirs["filename_adv_dir"],
    )
    # ckpt_dvec_cls_shadow, _ = create_checkpoint_dir(
    #     args,
    #     filenames_shadow_dvecs_cls_and_dirs["filename"],
    #     filenames_shadow_dvecs_cls_and_dirs["filename_dir"],
    # )

    ckpt_dvec_cls, _ = create_dp_checkpoint_dir(
        args,
        filenames_dvecs_cls_and_dirs["filename_adv"],
        filenames_dvecs_cls_and_dirs_dp["filename"],
        filenames_dvecs_cls_and_dirs["filename_adv_dir"],
        filenames_dvecs_cls_and_dirs_dp["filename_dir"],
    )
    # ckpt_dvec_cls, _ = create_checkpoint_dir(
    #     args,
    #     filenames_dvecs_cls_and_dirs["filename_adv"],
    #     filenames_dvecs_cls_and_dirs["filename_adv_dir"],
    # )

    ckpt_memb_inf_cls, _ = create_checkpoint_dir(
        args,
        filenames_and_dirs_memb_inf["filename"],
        filenames_and_dirs_memb_inf["filename_dir"],
    )

    # Model
    model_collection = {
        "memb_attk_model": MembInfAttk_v3,
        "shadow_model": DvecAttentivePooledClsE2E_v3,
        "model": DvecAttentivePooledClsE2E_v3,
    }

    attk_props = {
        "name": CWSpkIDE2E,
        "cost": NonTargetedCostCW(device),
        "tgt_label": NonTargetLabelStrategy(device),
        "attk_type": "CW",
    }

    loss_props = {"loss": nn.BCEWithLogitsLoss}

    eval_memb_inf_performance_on_dp_model_e2e_relativistic(
        args,
        hparams,
        device,
        model_collection,
        loss_props,
        attk_props,
        ckpt_dvec_cls_shadow,
        ckpt_dvec_cls,
        utts_counts_max,
        ckpt_memb_inf_cls,
    )


if __name__ == "__main__":
    main()
