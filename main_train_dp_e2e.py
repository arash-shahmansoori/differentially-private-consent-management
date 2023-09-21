import torch

from torch.optim import Adam
from sub_sampling import CreateSubSamples
from utils_e2e import (
    parse_args,
    HyperParamsDP,
    create_checkpoint_dir,
    create_dp_checkpoint_dir,
    create_filenames_e2e,
    create_filenames_e2e_shadow,
    create_filenames_dp_e2e,
    DvecAttentivePooledClsE2E,
    DvecAttentivePooledClsE2E_v3,
    DvecAttentivePooledClsTransformerE2E,
    ShadowDvecAttentivePooledClsE2E,
    GaussianSGD,
    DP_SGD_eps_list_at_delta,
)
from training_functions_e2e import (
    train_with_dp_e2e_full,
    train_with_dp_e2e_selective,
    train_with_dp_e2e_BiTFiT,
)
from attack_strategy_e2e import CWSpkIDE2E, NonTargetedCostCW, NonTargetLabelStrategy
from preprocess_data import ClassificationDatasetGdrSpkr, create_dataset_arguments_bkt

from opacus import PrivacyEngine

# from opacus.accountants.utils import get_noise_multiplier
from prv_accountant import Accountant
from tensorflow_privacy.privacy.analysis.gdp_accountant import compute_eps_poisson


def main():
    args = parse_args()
    hparams = HyperParamsDP()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    dp_mode = "selective"  # mode for dp fine-tuning, e.g., full, selective, BiTFiT

    # Select the appropriate utts percentage based on max utts counter
    utts_counts_max = 5

    # Create dataset
    data_dir_test, speaker_infos_test = create_dataset_arguments_bkt(
        args,
        hparams.buckets_dp,
        args.dp_data_dir_test,
        utts_counts_max,
    )
    dataset_test = ClassificationDatasetGdrSpkr(
        data_dir_test,
        speaker_infos_test,
        args.n_utterances_labeled,
        args.seg_len,
    )

    # Sub-sample
    sub_sample = CreateSubSamples(args)

    # Number of utterances per composite bucket
    utts_per_spk = sub_sample.num_per_spk_utts_progressive_mem(
        [len(hparams.buckets_dp) * args.spk_per_bucket]
    )

    indx_comp_init = sub_sample.utt_index_per_bucket_collection(
        [len(hparams.buckets_dp) * args.spk_per_bucket],
        utts_per_spk,
    )

    bs = len(indx_comp_init[0])
    dataset_length = 1 * len(dataset_test)
    sample_rate = bs / dataset_length

    num_points = int(
        (0.4 * (1 / sample_rate)) ** 2
    )  # The Edgeworth paper sample_rate:= 0.4/sqrt(num_points)
    points = [num_points]

    num_iter = 2

    # Edgeworth:
    # sgd = GaussianSGD(sigma=hparams.sigma, p=sample_rate, order=1)
    # eps_ew_est = [
    #     sgd.approx_eps_from_delta_edgeworth(
    #         hparams.delta,
    #         n,
    #         method="estimate",
    #     )
    #     for n in points
    # ]
    eps_ew_est2 = DP_SGD_eps_list_at_delta(
        hparams.sigma,
        points,
        sample_rate,
        hparams.delta,
        order=2,
    )
    # eps_ew_est3 = DP_SGD_eps_list_at_delta(
    #     hparams.sigma,
    #     points,
    #     sample_rate,
    #     hparams.delta,
    #     order=3,
    # )

    # # FFT:
    # accountant = Accountant(
    #     noise_multiplier=hparams.sigma,
    #     sampling_probability=sample_rate,
    #     delta=hparams.delta,
    #     eps_error=0.1,
    #     max_compositions=points[0],
    # )

    # results = [accountant.compute_epsilon(num_compositions=pt) for pt in points]
    # eps_low, eps_fft, eps_upper = (
    #     [item[0] for item in results],
    #     [item[1] for item in results],
    #     [item[2] for item in results],
    # )

    # GDP:
    # eps_gdp = [
    #     compute_eps_poisson(
    #         num_iter * pt / num_iter,
    #         hparams.sigma,
    #         dataset_length,
    #         bs,
    #         hparams.delta,
    #     )
    #     for pt in points
    # ]

    # print(eps_low, eps_fft, eps_upper)
    eps = eps_ew_est2[0][0]

    print(f"(ε = {eps:.3f}, δ = {hparams.delta}), num_iter:{num_iter}")

    if dp_mode == "full":
        # Model
        model_collection = {
            "model": DvecAttentivePooledClsE2E_v3,
            "shadow_model": ShadowDvecAttentivePooledClsE2E,
        }

        # Privacy engine
        privacy_engine = PrivacyEngine()
        # sigma = get_noise_multiplier(
        #     target_epsilon=hparams.eps,
        #     target_delta=hparams.delta,
        #     sample_rate=sample_rate,
        #     epochs=num_iter,
        # )

        filenames_dvecs_cls_and_dirs = create_filenames_e2e(
            hparams.buckets,
            dp_mode,
            utts_counts_max,
            args,
        )

        privacy_props = {
            "bs": bs,
            "privacy_engine": privacy_engine,
            "points": points,
            "sigma": hparams.sigma,
            "eps": eps,
            "delta": hparams.delta,
            "dataset": dataset_test,
            "len_dataset": dataset_length,
            "num_iter": num_iter,
            "utts_per_spk": utts_per_spk,
            "compute_eps_gdp": compute_eps_poisson,
            # "compute_eps_fft": accountant,
            "compute_eps_ews": DP_SGD_eps_list_at_delta,
        }

        # Training
        train = train_with_dp_e2e_full
    elif dp_mode == "selective":
        # Model
        model_collection = {
            "model": DvecAttentivePooledClsE2E,
            "shadow_model": ShadowDvecAttentivePooledClsE2E,
            "excluded_layer": "lstm",
        }

        # Privacy engine
        if "nonDP" not in args.clipping_mode:
            # sigma = get_noise_multiplier(
            #     target_epsilon=hparams.eps,
            #     target_delta=hparams.delta,
            #     sample_rate=sample_rate,
            #     epochs=num_iter,
            # )

            if "BK" in args.clipping_mode:
                clipping_mode = args.clipping_mode[3:]
            else:
                clipping_mode = "ghost"

        filenames_dvecs_cls_and_dirs = create_filenames_e2e(
            hparams.buckets,
            dp_mode,
            utts_counts_max,
            args,
        )

        privacy_props = {
            "bs": bs,
            "sigma": hparams.sigma,
            "eps": eps,
            "delta": hparams.delta,
            "points": points,
            "dataset": dataset_test,
            "len_dataset": dataset_length,
            "num_iter": num_iter,
            "utts_per_spk": utts_per_spk,
            "clipping_mode": clipping_mode,
            "compute_eps_gdp": compute_eps_poisson,
            # "compute_eps_fft": accountant,
            "compute_eps_ews": DP_SGD_eps_list_at_delta,
        }

        # Training
        train = train_with_dp_e2e_selective
    elif dp_mode == "BiTFiT":
        # Model
        model_collection = {
            "model": DvecAttentivePooledClsTransformerE2E,
            # "model": DvecAttentivePooledClsE2E_v3,
            # "shadow_model": ShadowDvecAttentivePooledClsE2E,
        }

        # Privacy engine
        if "nonDP" not in args.clipping_mode:
            # sigma = get_noise_multiplier(
            #     target_epsilon=hparams.eps,
            #     target_delta=hparams.delta,
            #     sample_rate=sample_rate,
            #     epochs=num_iter,
            # )

            if "BK" in args.clipping_mode:
                clipping_mode = args.clipping_mode[3:]
            else:
                clipping_mode = "ghost"

        # Privacy engine
        # privacy_engine = PrivacyEngine()
        # sigma = get_noise_multiplier(
        #     target_epsilon=hparams.eps,
        #     target_delta=hparams.delta,
        #     sample_rate=sample_rate,
        #     epochs=num_iter,
        # )

        filenames_dvecs_cls_and_dirs = create_filenames_e2e(
            hparams.buckets,
            "full",
            utts_counts_max,
            args,
        )

        privacy_props = {
            "bs": bs,
            # "privacy_engine": privacy_engine,
            "sigma": hparams.sigma,
            "eps": eps,
            "delta": hparams.delta,
            "dataset": dataset_test,
            "points": points,
            "len_dataset": dataset_length,
            "num_iter": num_iter,
            "utts_per_spk": utts_per_spk,
            "clipping_mode": clipping_mode,
            "compute_eps_gdp": compute_eps_poisson,
            # "compute_eps_fft": accountant,
            "compute_eps_ews": DP_SGD_eps_list_at_delta,
        }

        # Training
        train = train_with_dp_e2e_BiTFiT

    # Optimizer
    opt = {"opt": Adam}

    filenames_shadow_dvecs_cls_and_dirs = create_filenames_e2e_shadow(
        hparams.buckets,
        utts_counts_max,
        args,
    )

    filenames_dvecs_cls_and_dirs_dp = create_filenames_dp_e2e(
        hparams.buckets_dp,
        dp_mode,
        utts_counts_max,
        eps,
        hparams.delta,
        args,
    )

    ckpt_dvec_cls_shadow, _ = create_checkpoint_dir(
        args,
        filenames_shadow_dvecs_cls_and_dirs["filename"],
        filenames_shadow_dvecs_cls_and_dirs["filename_dir"],
    )

    ckpt_dp_dvec_cls, status_dp_dvec_cls = create_dp_checkpoint_dir(
        args,
        filenames_dvecs_cls_and_dirs["filename_adv"],
        filenames_dvecs_cls_and_dirs_dp["filename"],
        filenames_dvecs_cls_and_dirs["filename_adv_dir"],
        filenames_dvecs_cls_and_dirs_dp["filename_dir"],
    )
    # ckpt_dp_dvec_cls, status_dp_dvec_cls = None, None

    attk_props = {
        "name": CWSpkIDE2E,
        "cost": NonTargetedCostCW(device),
        "tgt_label": NonTargetLabelStrategy(device),
        "attk_type": "CW",
    }

    train(
        args,
        hparams,
        device,
        sub_sample,
        model_collection,
        opt,
        privacy_props,
        attk_props,
        ckpt_dvec_cls_shadow,
        ckpt_dp_dvec_cls,
        status_dp_dvec_cls,
        dp_mode,
        utts_counts_max,
        filenames_dvecs_cls_and_dirs_dp,
    )


if __name__ == "__main__":
    main()
