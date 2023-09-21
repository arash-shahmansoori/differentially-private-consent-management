import os
import re
import json


import torch
import librosa

import pandas as pd
import numpy as np

from pathlib import Path
from uuid import uuid4
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from preprocess_data import Wav2Mel


def extract_features(audio, args):
    """Returns a np.array with size (args.feature_dim,'n')
    where n is the number of audio frames."""

    yt, _ = librosa.effects.trim(audio, top_db=args.top_db)
    yt = normalize(yt)
    ws = int(args.sample_rate * 0.001 * args.window_size)
    st = int(args.sample_rate * 0.001 * args.stride)

    if args.feature == "fbank":
        feat = librosa.feature.melspectrogram(
            y=audio,
            sr=args.sample_rate,
            n_mels=args.feature_dim,
            n_fft=ws,
            hop_length=st,
        )
        feat = np.log(feat + 1e-6)
    elif args.feature == "mfcc":
        feat = librosa.feature.mfcc(
            y=audio, sr=args.sample_rate, n_mfcc=args.feature_dim
        )
    else:
        raise ValueError("Unsupported Acoustic Feature: " + args.feature)

    feat = [feat]
    if args.delta:
        feat.append(librosa.feature.delta(feat[0]))
    if args.delta_delta:
        feat.append(librosa.feature.delta(feat[0], order=2))
    feat = np.concatenate(feat, axis=0)
    return feat


def normalize(yt):
    yt_max = np.max(yt)
    yt_min = np.min(yt)
    a = 1.0 / (yt_max - yt_min)
    b = -(yt_max + yt_min) / (2 * (yt_max - yt_min))

    yt = yt * a + b
    return yt


def dp_logics(utts_counts, dp_mode, utts_counts_max):
    """Limit the number of utterances as follows for dp.

    Args:
        - utts_counts: the current utterance index
        - dp_mode: the dp mode
        - utts_counts_max: the max utts counts

    Returns:
        dp_logic[dp_mode]: logics for a specific key ``dp_mode''.
    """

    dp_logic = {
        "train": utts_counts
        <= utts_counts_max,  # first portion of utterances used for training
        "test": utts_counts
        > utts_counts_max,  # second portion of utterances used for dp-finetuning
    }

    return dp_logic[dp_mode]


class DisjointTrainTestCounter(Dataset):
    def __init__(self, args, root, filename):
        self.args = args

        df_train = pd.DataFrame(columns=["speaker_id", "wave"])

        i = 0
        for path, _, files in os.walk(root):
            for name in files:
                path = path.replace("\\", "/")
                speaker_id = path.split("/")[-2]

                if name.endswith(".flac"):
                    name_a = name.split(".")
                    name_b = name_a[0].split("-")

                    wave, sample_rate = librosa.load(os.path.join(path, name), sr=16000)

                    df_train.loc[i] = [speaker_id] + [wave]
                    i += 1

        labels_train = pd.DataFrame(columns=["speaker_id", "gender"])

        f = open(filename, "r", encoding="utf8").readlines()

        i = 0
        for idx, line in enumerate(f):
            if idx > 11:
                parsed = re.split("\s+", line)
                if (
                    parsed[4] == "dev-clean"
                    or parsed[4] == f"train-clean-100"
                    or parsed[4] == f"train-other-500"
                    or parsed[4] == "dev-other"
                    or parsed[4] == "test-other"
                ):
                    labels_train.loc[i] = (
                        parsed[0],
                        parsed[2],
                    )  # speaker_id and label (M/F)
                    i += 1

        dataset_train = pd.merge(
            df_train, labels_train, on="speaker_id"
        )  # merging the two dataframes on 'speaker_id' for training.

        self.samples_train = dataset_train

        self.gender_train_list = sorted(set(dataset_train["gender"]))
        self.speaker_train_list = sorted(set(self.samples_train["speaker_id"]))

    def __getitem__(self, i):
        sample_train = self.samples_train

        gdr_train = sample_train["gender"][i]
        wave_train = sample_train["wave"][i]
        spk_train = sample_train["speaker_id"][i]

        feature_train = extract_features(wave_train, self.args).swapaxes(0, 1)

        # zero mean and unit variance
        feature_train = (feature_train - feature_train.mean()) / feature_train.std()

        return spk_train, gdr_train, feature_train

    def __len__(self):
        return len(self.samples_train)


class DatasetPartial(Dataset):
    def __init__(
        self,
        args,
        root,
        filename,
        selected_spks_bkts,
        dp_mode,
        utts_counts_max,
    ):
        self.args = args

        df_train = pd.DataFrame(columns=["speaker_id", "wave"])

        i = 0
        for path, _, files in os.walk(root):
            for name in files:
                path = path.replace("\\", "/")
                speaker_id = path.split("/")[-2]

                if speaker_id in selected_spks_bkts:
                    if name.endswith(".flac"):
                        name_a = name.split(".")
                        name_b = name_a[0].split("-")

                        wave, _ = librosa.load(os.path.join(path, name), sr=16000)

                        pcnt_logic = dp_logics(int(name_b[2]), dp_mode, utts_counts_max)

                        if pcnt_logic:
                            df_train.loc[i] = [speaker_id] + [wave]
                            i += 1

        labels_train = pd.DataFrame(columns=["speaker_id", "gender"])

        f = open(filename, "r", encoding="utf8").readlines()

        i = 0
        for idx, line in enumerate(f):
            if idx > 11:
                parsed = re.split("\s+", line)
                if (
                    parsed[4] == "dev-clean"
                    or parsed[4] == f"train-clean-100"
                    or parsed[4] == f"train-other-500"
                    or parsed[4] == "dev-other"
                    or parsed[4] == "test-other"
                ):
                    if parsed[0] in selected_spks_bkts:
                        labels_train.loc[i] = (
                            parsed[0],
                            parsed[2],
                        )  # speaker_id and label (M/F)

                        i += 1

        dataset_train = pd.merge(
            df_train,
            labels_train,
            on="speaker_id",
        )  # merge two dataframes on 'speaker_id'

        self.samples_train = dataset_train

    def __getitem__(self, i):
        sample_train = self.samples_train

        gdr_train = sample_train["gender"][i]
        wave_train = sample_train["wave"][i]
        spk_train = sample_train["speaker_id"][i]

        feature_train = extract_features(wave_train, self.args).swapaxes(0, 1)

        # zero mean and unit variance
        feature_train = (feature_train - feature_train.mean()) / feature_train.std()

        return spk_train, gdr_train, feature_train

    def __len__(self):
        return len(self.samples_train)


class DatasetFull(Dataset):
    def __init__(
        self,
        args,
        root,
        filename,
        selected_spks_bkts,
    ):
        self.args = args

        df_train = pd.DataFrame(columns=["speaker_id", "wave"])

        i = 0
        for path, _, files in os.walk(root):
            for name in files:
                path = path.replace("\\", "/")
                speaker_id = path.split("/")[-2]

                if speaker_id in selected_spks_bkts:
                    if name.endswith(".flac"):
                        wave, _ = librosa.load(os.path.join(path, name), sr=16000)

                        df_train.loc[i] = [speaker_id] + [wave]
                        i += 1

        labels_train = pd.DataFrame(columns=["speaker_id", "gender"])

        f = open(filename, "r", encoding="utf8").readlines()

        i = 0
        for idx, line in enumerate(f):
            if idx > 11:
                parsed = re.split("\s+", line)
                if (
                    parsed[4] == "dev-clean"
                    or parsed[4] == f"train-clean-100"
                    or parsed[4] == f"train-other-500"
                    or parsed[4] == "dev-other"
                    or parsed[4] == "test-other"
                ):
                    if parsed[0] in selected_spks_bkts:
                        labels_train.loc[i] = (
                            parsed[0],
                            parsed[2],
                        )  # speaker_id and label (M/F)

                        i += 1

        dataset_train = pd.merge(
            df_train,
            labels_train,
            on="speaker_id",
        )  # merge two dataframes on 'speaker_id'

        self.samples_train = dataset_train

    def __getitem__(self, i):
        sample_train = self.samples_train

        gdr_train = sample_train["gender"][i]
        wave_train = sample_train["wave"][i]
        spk_train = sample_train["speaker_id"][i]

        feature_train = extract_features(wave_train, self.args).swapaxes(0, 1)

        # zero mean and unit variance
        feature_train = (feature_train - feature_train.mean()) / feature_train.std()

        return spk_train, gdr_train, feature_train

    def __len__(self):
        return len(self.samples_train)
