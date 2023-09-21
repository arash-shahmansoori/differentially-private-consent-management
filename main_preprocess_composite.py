import torch

from utils_e2e import parse_args, HyperParamsDP, cor_seq_counter_list
from create_disjoint_train_test_datasets import (
    preprocess_comp,
    DisjointTrainTestCounter,
    DatasetPartial,
)


def main():
    args = parse_args()
    hparams = HyperParamsDP()

    # Select the mode: train/test
    dp_dataset_mode = "train"

    # Select the appropriate utts percentage based on max atts counter
    utts_counts_max = 5

    if dp_dataset_mode == "train":
        output_dir = args.output_dir_dp_train  # For training
    elif dp_dataset_mode == "test":
        output_dir = args.output_dir_dp_test  # For testing

    # Clean utts
    root_name = (
        f"./data/LibriSpeech_modular/agnt_{args.agnt_num}_spks_{args.n_speakers}"
    )
    file_name = f"data/LibriSpeech_modular/agnt_{args.agnt_num}_spks_{args.n_speakers}/Speakers.txt"

    # Create buckets of speakers
    stride = args.spk_per_bucket
    labels = [i for i in range(args.n_speakers)]
    outputs = cor_seq_counter_list(len(labels), args.spk_per_bucket, stride)

    spks_bkts = (
        torch.tensor([outputs[bkt_id] for bkt_id in hparams.buckets]).view(-1).tolist()
    )

    dataset_counter = DisjointTrainTestCounter(
        args,
        root_name,
        file_name,
    )

    # Create selected speakers in the composite bucket
    speaker_train_list = sorted(
        list(map(lambda x: int(x), dataset_counter.speaker_train_list))
    )
    _selected_spks_bkts = [speaker_train_list[i] for i in spks_bkts]
    selected_spks_bkts = list(map(lambda x: str(x), _selected_spks_bkts))

    # Use this to obtain public/private portion of dataset from the composite bucket
    dataset = DatasetPartial(
        args,
        root_name,
        file_name,
        selected_spks_bkts,
        dp_dataset_mode,
        utts_counts_max,
    )

    preprocess_comp(
        args,
        output_dir,
        selected_spks_bkts,
        hparams.buckets,
        dataset,
        utts_counts_max,
    )


if __name__ == "__main__":
    main()
