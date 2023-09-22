# Composite-Bucket-Based Adversarial Training with Differential Private Fine-Tuning for Speaker Recognition in Voice Assistant Systems

This code-base supports differential privacy for speaker recognition in voice assistant systems using the private data and a small portion of publicly available data for the paper ``**Dynamic Recognition of Speakers for Consent Management by Contrastive Embedding Replay''** ([arXiv](https://arxiv.org/abs/2205.08459)).

In particular, it addresses the differential privacy (DP) for speaker recognition in voice assistant systems using the private data and a small portion of publicly available data for a composite bucket of speakers.

Definition: A composite bucket is comprised of a bucket or buckets of speakers.

The main goals are the following:

- Recognition of a set of speakers (in the composite bucket) with small portion of available public data
- Providing adversarial protection for those speakers
- Providing DP fine-tuning for private data of those speakers
- Providing defence against relativistic membership attack for both private & public data of those speakers after DP fine-tuning
- Providing minimal trade-offs in performance and accuracy

## Installation

Install the required packages for this repository by running the following script.

```angular2
pip install -r requirements.txt
```

## Data

To create speakers per agent use the following steps:

Make sure that the LibriSpeech dataset is downloaded and follows the following tree structure in the data folder.

```angular2
data ---.
        ¦---> LibriSpeech --> train-clean-100 --> Speakers' folders
                          --> Books.txt
                          --> Chapters.txt
                          --> License.txt
                          --> ReadMe.txt
                          --> Speakers.txt

```

Choose the appropriate root directory "root_dir" and the destination directory "dest_dir" in the "create_spks_per_agnt.py". For instance, the following can be used for the "LibriSpeech" dataset:

```angular2
root_dir = "data\\LibriSpeech"
dest_dir = f"data\\LibriSpeech_modular\\agnt_{args.agnt_num}_spks_{num_spks_per_agnt}"
```

Once the aformentioned directories are selected, use the following command to create speakers per agent.

```angular2
python create_spks_per_agnt.py
```

## Prepare Public/Private Data

To prepare public/private data per (composite) bucket for DP training follow the steps below.
Select the corresponding mode by setting the variable "dp_dataset_mode" to appropriate value.
Choose the appropriate "utts_counts_max" to create a portion of dataset for the corresponding mode.
Create a composite bucket of speakers as described in the "main_dp_preprocess.py", and run the following script:

```angular2
python main_preprocess_composite.py
```

## Training with Small Portion of Publicly Available Data

Train feature extraction and classifier networks on small portion of available public dataset.

To do so, run the following script:

```angular2
python main_train_composite_e2e.py
```

## CW Adversarial Samples

CW adversarial samples are created in the input space according to the pretrained checkpoints of the feature extraction and classification networks trained on public dataset.

## Adversarial Training

Using the CW adversarial samples obtained in the input space from the previous step, feature extraction and classification networks are adversarially trained from scratch using small portion of public data and corresponding adversarial samples.

To do so, run the following script:

```angular2
python main_train_adv_composite_e2e.py
```

## DP Fine-Tuning

Fine-tune feature extraction and classification networks on the private data with DP starting from adversarially trained checkpoints from the public dataset.

To do so, run the following script:

```angular2
python main_train_dp_e2e.py
```

*Note: the supported DP fine-tunings in this repository are:

- Full DP fine-tuning of all the layers (full)
- Bias term DP fine-tuning of all the layers (BiTFiT)
- Selective DP fine-tuning excluding the LSTM/Transformer layer (selective)

For the case of selective DP fine-tuning, the LSTM layer, trained using public data and corresponding adversarial samples, is freezed during the DP fine-tuning process on the private data.

## Train Relativistic Membership Attacker

Train relativistic membership attacker network on the publicly available samples together with the corresponding CW adversarial samples.

To do so, run the following script:

```angular2
python main_train_memb_inf_attk_e2e.py
```

## Checkpoints

The corresponding checkpoints in all different steps should be saved in the folder "checkpoints_e2e".

## Evaluate Relativistic Membership Attacker Performance

Evaluate the attacker performance on adversarially trained features on public data and DP fine-tuned on private data.

To do so, run the following script:

```angular2
python main_eval_dp_e2e.py
```

## Results

Save the metrics of interest, e.g., spent privacy and validation accuracy with the sub-folder names "pv_spent" and "val_acc_adv_dp", respectively, in the folder "results_e2e".

## References

The proposed method in this repository provides DP protection for the private data based on the bucket-based consent management in the following paper, accepted for publication in IEEE Transactions on Neural Networks and Learning Systems (TNNLS):

- [Dynamic Recognition of Speakers for Consent
Management by Contrastive Embedding Replay](https://arxiv.org/abs/2205.08459)

Note: The paper mentioned above has been recently accepted for publication with the "DOI:10.1109/TNNLS.2023.3317493" in IEEE Transactions on Neural Networks and Learning Systems (TNNLS). Please cite the paper using the following format in the future if you are using the current repository:

Arash Shahmansoori and Utz Roedig."Dynamic Recognition of Speakers for Consent
Management by Contrastive Embedding Replay." IEEE Transactions on Neural Networks and Learning Systems, vol. [Volume], no. [Issue], pp. [Page range], [Month] [2023]. DOI:10.1109/TNNLS.2023.3317493 (corresponding author: Arash Shahmansoori)

- Cite this repository

## License

[MIT License](LICENSE)

---
***Contact the Author***

The author "Arash Shahmansoori" e-mail address: <arash.mansoori65@gmail.com>
