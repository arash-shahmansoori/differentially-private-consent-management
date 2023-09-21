import torch
import numpy as np


class CreateSubSamples:
    """
    Create sub samples for the dp training.
    """

    def __init__(self, args):
        self.args = args

    def num_per_spk_utts_progressive_mem(self, spk_per_bucket_storage):
        total_spks_per_bkts_storage = 0
        for spk_per_bucket in spk_per_bucket_storage:
            total_spks_per_bkts_storage += spk_per_bucket

        utts_per_spk = torch.floor(
            torch.tensor((self.args.max_mem / (total_spks_per_bkts_storage)))
        )
        return int(utts_per_spk)

    def utt_index_per_bucket_collection(
        self,
        spk_per_bucket_storage,
        num_utts,
        prng=None,
    ):
        # Randomly selects "num_utts" utterances per speaker per bucket.
        prng = prng if prng else np.random

        lf_collection = []
        for spk_per_bucket in spk_per_bucket_storage:
            total_spk_per_bucket = spk_per_bucket

            l = [
                (
                    torch.from_numpy(
                        prng.choice(
                            range(
                                self.args.n_utterances_labeled * i,
                                self.args.n_utterances_labeled * (i + 1),
                            ),
                            num_utts,
                            replace=True,
                        )
                    ).int()
                ).tolist()
                for i in range(total_spk_per_bucket)
            ]
            lf = [u for s in l for u in s]

            lf_collection.append(lf)

        return lf_collection
