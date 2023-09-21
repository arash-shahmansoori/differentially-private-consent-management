"""Dataset for training from scratch."""

import random
from pathlib import Path
from typing import Union
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np


class DatasetScratch(Dataset):
    """Sample utterances from speakers."""

    def __init__(
        self,
        data_dir: Union[str, Path],
        speaker_infos: dict,
        n_utterances: int,
        seg_len: int,
    ):
        """
        Args:
            data_dir (string): path to the directory of pickle files.
            n_utterances (int): # of utterances per speaker to be sampled.
            seg_len (int): the minimum length of segments of utterances.
        """

        self.data_dir = data_dir
        self.n_utterances = n_utterances
        self.seg_len = seg_len
        self.speaker_infos = speaker_infos
        self.infos = []
        self.state = 0

        for spk_idx, uttr_infos in enumerate(self.speaker_infos.values()):
            feature_paths = [
                (uttr_info["feature_path"], uttr_info["gender"], spk_idx)
                for uttr_info in uttr_infos
                if uttr_info["mel_len"] > self.seg_len
            ]
            if len(feature_paths) > n_utterances:
                self.infos.append(feature_paths)

        self.infos_flatten = [i_ for s_ in self.infos for i_ in s_]

    def __len__(self):
        return len(self.infos_flatten)

    def __getitem__(self, index):
        # self.state += 1
        # print(self.state)
        feature_paths_tuple = self.infos_flatten[index]
        # feature_path_unpacked, gender_unpacked, speaker_unpacked = zip(
        #     *feature_paths_tuple
        # )
        feature_path_unpacked, gender_unpacked, speaker_unpacked = feature_paths_tuple

        uttr, gdr, spkr = (
            torch.load(Path(self.data_dir, feature_path_unpacked)),
            gender_unpacked,
            speaker_unpacked,
        )

        lefts = [random.randint(0, uttr.shape[0] - self.seg_len)]
        segments = uttr[lefts[0] : lefts[0] + self.seg_len, :]
        feat = segments
        feat = torch.stack(list(feat))
        gdr = self.gdr_to_label(gdr)
        return feat, spkr

    @staticmethod
    def gdr_to_label(gdr):
        new_gdr = list()
        for ind in range(len(gdr)):
            if gdr[ind] == "M":
                new_gdr.append(0)
            else:
                new_gdr.append(1)
        return new_gdr


def collateGdrSpkr(batch):
    """Collate a whole batch of utterances."""
    feat = [sample[0] for sample in batch]
    gender = [sample[1] for sample in batch]
    speaker = [sample[2] for sample in batch]
    flatten_gender = [item for s in gender for item in s]
    flatten_gender = torch.tensor(flatten_gender, dtype=torch.long)
    flatten_speaker = [item for s in speaker for item in s]
    flatten_speaker = torch.tensor(flatten_speaker, dtype=torch.long)
    flatten = [u for s in feat for u in s]
    flatten = pad_sequence(flatten, batch_first=True, padding_value=0)
    return flatten, flatten_gender, flatten_speaker
