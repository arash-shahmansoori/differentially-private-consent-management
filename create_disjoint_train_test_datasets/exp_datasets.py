import os
import re
import json

import pandas as pd
import numpy as np

import torch
import librosa

from pathlib import Path
from uuid import uuid4
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

from preprocess_data import Wav2Mel


def extract_features(audio, args):
    """Returns a np.array with size (args.feature_dim,'n') where n is the number of audio frames."""

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


def pcnt_logics(utts_counts, pcnt_old):
    """Limit the number of old utterances as follows.

    Args:
        - utts_counts: the number of utterances in different folders of dataset.
        - pcnt_old: the percentage of old utterances used
        for dynamic new registrations.

    Returns:
        pcnt_logic[pcnt_old]: percentage logics ``pcnt_logic'' for
        a specific key ``pcnt_old''.

    """

    pcnt_logic = {
        "ten": utts_counts > 7
        and utts_counts <= 11,  # for 10% use of old training utterances
        "twenty": utts_counts > 7
        and utts_counts <= 15,  # for 20% use of old training utterances
        "thirty": utts_counts > 7
        and utts_counts <= 20,  # for 30% use of old training utterances
        "forty": utts_counts > 7
        and utts_counts <= 25,  # for 40% use of old training utterances
        "fifty": utts_counts > 7
        and utts_counts <= 30,  # for 50% use of old training utterances
        "sixty": utts_counts > 7
        and utts_counts <= 35,  # for 60% use of old training utterances
        "seventy": utts_counts > 7
        and utts_counts <= 40,  # for 70% use of old training utterances
        "eighty": utts_counts > 7
        and utts_counts <= 45,  # for 80% use of old training utterances
        "ninty": utts_counts > 7
        and utts_counts <= 60,  # for 90% use of old training utterances
        "full": utts_counts > 7,  # for full use of old training utterances
        "eval": utts_counts <= 7,  # for use of testing/evaluation utterances
    }

    return pcnt_logic[pcnt_old]


class DisjointTrainTest(Dataset):
    def __init__(self, args, root, filename, pcnt_old):
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

                    pcnt_logic = pcnt_logics(int(name_b[2]), pcnt_old)

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


def preprocess(args, output_dir, root_name, file_name, pcnt_old):
    """Preprocess audio files into features for training."""

    output_dir_path = Path(f"{output_dir}_{args.agnt_num}")
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # wav2mel = Wav2Mel(args)

    # torch.save(wav2mel, str(output_dir_path / "wav2mel.pt"))

    dataset = DisjointTrainTest(args, root_name, file_name, pcnt_old)

    print(len(dataset))

    dataloader = DataLoader(dataset, batch_size=1)

    infos_gender_speaker = {
        "n_mels": args.feature_dim,
        "speaker_gender": {
            speaker_name: [] for speaker_name in dataset.speaker_train_list
        },
    }

    for speaker_name, gender_name, mel_tensor in tqdm(dataloader):
        speaker_name = speaker_name[0]
        gender_name = gender_name[0]

        mel_tensor = mel_tensor.squeeze(0)

        random_file_path = output_dir_path / f"uttr-{uuid4().hex}.pt"
        # torch.save(mel_tensor, random_file_path)

        infos_gender_speaker["speaker_gender"][speaker_name].append(
            {
                "feature_path": random_file_path.name,
                "mel_len": mel_tensor.shape[0],
                "gender": gender_name,
            }
        )

        # with open(output_dir_path / "metadata.json", "w") as f:
        #     json.dump(infos_gender_speaker, f, indent=2)
