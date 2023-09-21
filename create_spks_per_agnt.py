from pathlib import Path
from utils_e2e import parse_args, create_spks_per_agnt_dataset

args = parse_args()


num_spks_per_agnt = args.n_speakers


root_dir = "data\\LibriSpeech"
dest_dir = f"data\\LibriSpeech_modular\\agnt_{args.agnt_num}_spks_{num_spks_per_agnt}"


dest_dir_agnt = Path(dest_dir)

if not dest_dir_agnt.exists():
    print(f"The directory does not exist and will be created.")
    dest_dir_agnt.mkdir(parents=True, exist_ok=True)


create_spks_per_agnt_dataset(root_dir, dest_dir_agnt, args.agnt_num, num_spks_per_agnt)
