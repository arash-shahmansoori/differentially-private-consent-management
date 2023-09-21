import librosa
import torch
import torch.nn as nn
import numpy as np

# from .utils import parse_args
# from .infinite_dataloader import infinite_iterator


class ExtractFeatures:
    def __init__(self, args):
        self.args = args

    def normalize(self, yt):
        yt_max = np.max(yt)
        yt_min = np.min(yt)
        a = 1.0 / (yt_max - yt_min)
        b = -(yt_max + yt_min) / (2 * (yt_max - yt_min))
        yt = yt * a + b
        return yt

    def forward(self, audio):
        yt, _ = librosa.effects.trim(audio, top_db=self.args.top_db)
        yt = self.normalize(yt)
        ws = int(self.args.sample_rate * 0.001 * self.args.window_size)
        st = int(self.args.sample_rate * 0.001 * self.args.stride)

        if self.args.feature == "fbank":
            feat = librosa.feature.melspectrogram(
                y=audio,
                sr=self.args.sample_rate,
                n_mels=self.args.feature_dim,
                n_fft=ws,
                hop_length=st,
            )
        feat = np.log(feat + 1e-6)
        # elif self.args.feature == "mfcc":
        #     feat = librosa.feature.mfcc(
        #     y=audio, sr=self.args.sample_rate, n_mfcc=self.args.feature_dim
        # )
        # else:
        #     raise ValueError("Unsupported Acoustic Feature: " + self.args.feature)

        feat = [feat]
        if self.args.delta:
            feat.append(librosa.feature.delta(feat[0]))
        if self.args.delta_delta:
            feat.append(librosa.feature.delta(feat[0], order=2))
        feat = np.concatenate(feat, axis=0)
        return feat  # returns a np.array with size (40,'n') where n is the number of audio frames.


class Wav2Mel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.feat_extract = ExtractFeatures(args)

    def forward(self, wav_tensor):
        mel_spec = self.feat_extract.forward(wav_tensor)
        return mel_spec
