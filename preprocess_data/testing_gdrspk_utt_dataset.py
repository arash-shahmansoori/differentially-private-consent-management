"""Dataset for speaker embedding."""

import random
from pathlib import Path
from typing import Union
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np


class ClassificationDatasetGdrSpkr_(Dataset):
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
        # self.state = 0

        for spk_idx, uttr_infos in enumerate(self.speaker_infos.values()):
            feature_paths = [
                (
                    uttr_info["feature_path"],
                    uttr_info["gender"],
                    spk_idx,
                    uttr_info["utt_index"],
                )
                for uttr_info in uttr_infos
                if uttr_info["mel_len"] > self.seg_len
            ]
            if len(feature_paths) > n_utterances:
                self.infos.append(feature_paths)

        # print(len(self.infos[2]))

        self.infos_flatten = [i_ for s_ in self.infos for i_ in s_]
        # print(len(self.infos_flatten))

    def __len__(self):
        # # self.infos_flatten = [i_ for s_ in self.infos for i_ in s_]
        # len_ = 0
        # for i in range(5):
        #     len_ += len(self.infos[i])

        # return len_
        # # return len(self.infos_flatten)
        return len(self.infos)

    def __getitem__(self, index):
        # self.state += 1
        # print(self.state)
        feature_paths_tuple = random.sample(self.infos[index], self.n_utterances)
        feature_path_unpacked, gender_unpacked, speaker_unpacked, utt_unpacked = zip(
            *feature_paths_tuple
        )
        uttrs_spkrs_gdrs = [
            (torch.load(Path(self.data_dir, feature_path)), gdr, spkr, utt_)
            for feature_path, gdr, spkr, utt_ in zip(
                list(feature_path_unpacked),
                list(gender_unpacked),
                list(speaker_unpacked),
                list(utt_unpacked),
            )
        ]
        lefts = [
            (
                random.randint(0, uttr[0].shape[0] - self.seg_len),
                uttr[1],
                uttr[2],
                uttr[3],
            )
            for uttr in uttrs_spkrs_gdrs
        ]
        segments = [
            (uttr[0][left[0] : left[0] + self.seg_len, :], uttr[1], uttr[2], uttr[3])
            for uttr, left in zip(uttrs_spkrs_gdrs, lefts)
        ]
        feat, gdr, spk, utts_ = zip(*segments)
        feat = torch.stack(list(feat))
        gdr = self.gdr_to_label(gdr)
        t = 0
        return feat, gdr, spk, utts_

    @staticmethod
    def gdr_to_label(gdr):
        new_gdr = list()
        for ind in range(len(gdr)):
            if gdr[ind] == "M":
                new_gdr.append(0)
            else:
                new_gdr.append(1)
        return new_gdr


def collateGdrSpkr_(batch):
    """Collate a whole batch of utterances."""
    feat = [sample[0] for sample in batch]
    gender = [sample[1] for sample in batch]
    speaker = [sample[2] for sample in batch]
    utt_indx = [sample[3] for sample in batch]
    flatten_gender = [item for s in gender for item in s]
    flatten_gender = torch.tensor(flatten_gender, dtype=torch.long)
    flatten_speaker = [item for s in speaker for item in s]
    flatten_uttr = [item for s in utt_indx for item in s]
    flatten_speaker = torch.tensor(flatten_speaker, dtype=torch.long)
    flatten_uttr = torch.tensor(flatten_uttr, dtype=torch.long)
    flatten = [u for s in feat for u in s]
    flatten = pad_sequence(flatten, batch_first=True, padding_value=0)
    return flatten, flatten_gender, flatten_speaker, flatten_uttr


class ReducedDataset_(Dataset):
    """To reduce a dataset, taking only samples corresponding to provided indeces.
    This is useful for splitting a dataset into a training and validation set."""

    def __init__(self, original_dataset, indeces):
        super().__init__()
        self.dataset = original_dataset
        self.indeces = indeces

    def __len__(self):
        return len(self.indeces)

    def __getitem__(self, index):
        return self.dataset[self.indeces[index]]


class SubDatasetGdrSpk_(Dataset):
    """To sub-sample a dataset, taking only those samples with label in [sub_labels].

    After this selection of samples has been made, it is possible to transform the target-labels,
    which can be useful when doing continual learning with fixed number of output units."""

    def __init__(self, original_dataset, sub_labels_spk):
        super().__init__()
        self.dataset = original_dataset
        self.sub_indeces = []

        for index in range(len(self.dataset)):
            label_spk = np.unique(self.dataset[index][2])
            if label_spk in sub_labels_spk:
                self.sub_indeces.append(index)

    def __len__(self):
        return len(self.sub_indeces)

    def __getitem__(self, index):
        sample = self.dataset[self.sub_indeces[index]]
        return sample


class TransformedDataset_(Dataset):
    """To modify an existing dataset with a transform.
    This is useful for creating different permutations without loading the data multiple times."""

    def __init__(self, original_dataset, target_transform=None):
        super().__init__()
        self.dataset = original_dataset
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        input, target = self.dataset[index]
        if self.target_transform:
            #     targets = []
            #     for t in target:
            # target = tuple(target)
            target = self.target_transform(target)
        #         targets.append(target)
        return (input, target)
