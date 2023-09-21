import json
import torch

from pathlib import Path
from uuid import uuid4
from tqdm import tqdm

from torch.utils.data import DataLoader
from preprocess_data import Wav2Mel


def preprocess_comp(
    args,
    output_dir,
    selected_spks_bkts,
    bucket,
    dataset,
    utts_counts_max,
):
    """Preprocess audio files into features for training."""

    output_dir_path = Path(
        f"{output_dir}_bkt_{bucket}_max_utts_count_{utts_counts_max}_agnt_{args.agnt_num}"
    )
    output_dir_path.mkdir(parents=True, exist_ok=True)

    wav2mel = Wav2Mel(args)

    torch.save(wav2mel, str(output_dir_path / "wav2mel.pt"))

    dataloader = DataLoader(dataset, batch_size=1)

    infos_gender_speaker = {
        "n_mels": args.feature_dim,
        "speaker_gender": {speaker_name: [] for speaker_name in selected_spks_bkts},
    }

    for speaker_name, gender_name, mel_tensor in tqdm(dataloader):
        speaker_name = speaker_name[0]
        gender_name = gender_name[0]

        mel_tensor = mel_tensor.squeeze(0)

        random_file_path = output_dir_path / f"uttr-{uuid4().hex}.pt"
        torch.save(mel_tensor, random_file_path)

        infos_gender_speaker["speaker_gender"][speaker_name].append(
            {
                "feature_path": random_file_path.name,
                "mel_len": mel_tensor.shape[0],
                "gender": gender_name,
            }
        )

        with open(output_dir_path / "metadata.json", "w") as f:
            json.dump(infos_gender_speaker, f, indent=2)
