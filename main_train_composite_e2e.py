import torch

from torch.optim import Adam
from utils_e2e import (
    parse_args,
    HyperParamsDP,
    create_filenames_e2e_shadow,
    create_filenames_e2e,
    create_checkpoint_dir,
    DvecAttentivePooledClsE2E,
    DvecAttentivePooledClsE2E_v3,
    DvecAttentivePooledClsTransformerE2E,
    ShadowDvecAttentivePooledClsE2E,
)
from training_functions_e2e import train_comp_e2e


def main():
    args = parse_args()
    hparams = HyperParamsDP()

    mode = "selective"

    # Select the appropriate utts percentage based on max utts counter
    utts_counts_max = 5

    # Specify the device to run the simulations on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    # Optimizer
    opt = {"opt": Adam}

    # Model and filenames
    if mode == "full":
        model_collection = {"model": DvecAttentivePooledClsE2E_v3}

        filenames_dvecs_cls_and_dirs = create_filenames_e2e(
            hparams.buckets,
            mode,
            utts_counts_max,
            args,
        )
    elif mode == "selective":
        model_collection = {"model": DvecAttentivePooledClsTransformerE2E}

        # filenames_dvecs_cls_and_dirs = create_filenames_e2e(
        #     hparams.buckets,
        #     mode,
        #     utts_counts_max,
        #     args,
        # )
    elif mode == "BiTFiT":
        model_collection = {"model": DvecAttentivePooledClsE2E_v3}

        filenames_dvecs_cls_and_dirs = create_filenames_e2e(
            hparams.buckets,
            mode,
            utts_counts_max,
            args,
        )
    else:
        model_collection = {"model": ShadowDvecAttentivePooledClsE2E}

        filenames_dvecs_cls_and_dirs = create_filenames_e2e_shadow(
            hparams.buckets,
            utts_counts_max,
            args,
        )

    # Set ``ckpt_cls'' to the available checkpoint
    # ckpt_cls, _ = create_checkpoint_dir(
    #     args,
    #     filenames_dvecs_cls_and_dirs["filename"],
    #     filenames_dvecs_cls_and_dirs["filename_dir"],
    # )

    train_comp_e2e(
        args,
        hparams,
        device,
        model_collection,
        opt,
        # ckpt_cls,
        None,
        utts_counts_max,
        # filenames_dvecs_cls_and_dirs,
        None,
    )


if __name__ == "__main__":
    main()
