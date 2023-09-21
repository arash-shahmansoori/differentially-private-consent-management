import argparse


def boolean_string(s):
    if s not in {"False", "True"}:
        raise ValueError("Not a valid boolean string")
    return s == "True"


def parse_args():
    # Commandline arguments
    parser = argparse.ArgumentParser(
        description="Continual Learning with Contrastive Embedding Buffer"
    )
    ################################ Acoustic feature parameters ###########################
    parser.add_argument("--feature", default="fbank", type=str, help="acoustic feature")
    parser.add_argument(
        "--sample_rate", default=16000, type=int, help="sample rate of audio signal"
    )
    parser.add_argument(
        "--top_db", default=20, type=int, help="voice acticity detection"
    )
    parser.add_argument("--window_size", default=25, type=int, help="window size in ms")
    parser.add_argument("--stride", default=10, type=int, help="stride size")
    parser.add_argument(
        "--feature_dim", default=40, type=int, help="input acoustic feature dimension"
    )
    parser.add_argument(
        "--delta", default=False, type=bool, help="acoustic delta feature"
    )
    parser.add_argument(
        "--delta_delta", default=False, type=bool, help="acoustic d2elta feature"
    )
    ######################## Model #############################
    parser.add_argument("--dim_emb", default=256, type=int)
    parser.add_argument("--dim_emb_backend", default=128, type=int)
    parser.add_argument("--latent_dim", default=64, type=int)
    parser.add_argument("-n", "--n_speakers", type=int, default=40)
    parser.add_argument("--dim_cell", default=256, type=int)
    parser.add_argument("--num_layers", default=3, type=int)
    parser.add_argument("--e_dim", default=32, type=int)
    parser.add_argument("--hidden_dim", default=128, type=int)
    parser.add_argument("--gp_norm_dvector", default=4, type=int)
    parser.add_argument("--gp_norm_cls", default=2, type=int)
    ######################## Optimizer ##########################
    parser.add_argument(
        "--epoch",
        dest="epoch",
        default=100,
        type=int,
        help="The number of epochs used for one bucket. (default: %(default)s)",
    )
    parser.add_argument(
        "--epoch_test",
        dest="epoch_test",
        default=10,
        type=int,
        help="The number of epochs used for one bucket. (default: %(default)s)",
    )
    parser.add_argument(
        "--jl_dim", type=int, default=30, help="Training with differential privacy"
    )
    parser.add_argument("--clipping_mode", default="BK-MixOpt", type=str)
    parser.add_argument("--origin_params", nargs="+", default=None)
    ######################## Data #########################
    parser.add_argument("--data_dir", type=str, default="train_dir_agnt")
    parser.add_argument("--dp_data_dir_train", type=str, default="data\\dp_train")
    parser.add_argument("--dp_data_dir_test", type=str, default="data\\dp_test")
    parser.add_argument("--dp_data_dir_eval", type=str, default="data\\dp_eval")
    parser.add_argument("--data_dir_aug", type=str, default="train_dir_aug/")
    parser.add_argument("--test_data_dir", type=str, default="test_dir/")
    parser.add_argument("--test_data_dir_aug", type=str, default="test_dir_aug/")
    parser.add_argument("--validation_data_dir", type=str, default="test_dir_agnt")
    parser.add_argument("--test_data_dir_other", type=str, default="test_dir_other/")
    parser.add_argument("--validation_data_dir_aug", type=str, default="test_dir_aug/")
    parser.add_argument("-agt_num", "--agnt_num", type=int, default=3)
    parser.add_argument("-m_labeled", "--n_utterances_labeled", type=int, default=5)
    parser.add_argument("-t_labeled", "--nt_utterances_labeled", type=int, default=6)
    parser.add_argument("-v_labeled", "--nv_utterances_labeled", type=int, default=6)
    parser.add_argument("-n_utts_select", "--n_utts_selected", type=int, default=5)
    parser.add_argument("--seg_len", type=int, default=160)
    parser.add_argument("--spk_per_bucket", type=int, default=5)
    parser.add_argument("--stride_per_bucket", type=int, default=5)
    parser.add_argument("--valid_every", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=1)
    ######################## ER #########################
    parser.add_argument(
        "--max_mem",
        dest="max_mem",
        default=10 * 5,
        type=int,
        help="Memory buffer size (default: %(default)s)",
    )
    ################################## Checkpoints ####################################
    parser.add_argument(
        "-cp_d", "--checkpoint_dir", type=str, default="checkpoints_e2e"
    )
    ################################### Logging #######################################
    parser.add_argument("--log_training", default=True, type=bool)
    #################### Early Stopping ######################
    parser.add_argument(
        "--early_stopping",
        dest="early_stopping",
        default=True,
        type=boolean_string,
        help="To use the early stopping",
    )
    parser.add_argument(
        "--verbose_stopping",
        dest="verbose_stopping",
        default=False,
        type=boolean_string,
        help="Print the early stopping message during iterations",
    )
    parser.add_argument(
        "--min_delta",
        dest="min_delta",
        default=0,
        type=float,
        help="A minimum increase in the score to qualify as an improvement",
    )
    parser.add_argument(
        "--patience_stopping",
        dest="patience_stopping",
        default=5,
        type=int,
        help="Number of events to wait if no improvement and then stop the training.",
    )
    parser.add_argument(
        "--threshold_val_acc",
        dest="threshold_val_acc",
        default=96,
        type=int,
        help="Threshold validation accuracy for early stopping.",
    )
    parser.add_argument(
        "--threshold_val_acc_dp",
        dest="threshold_val_acc_dp",
        default=94,
        type=int,
        help="Threshold validation accuracy for early stopping with dp.",
    )
    ############################# Output path for data ############################
    parser.add_argument(
        "-o_dp_train", "--output_dir_dp_train", type=str, default="data\\dp_train"
    )
    parser.add_argument(
        "-o_dp_test", "--output_dir_dp_test", type=str, default="data\\dp_test"
    )
    parser.add_argument(
        "-o_dp_eval", "--output_dir_dp_eval", type=str, default="data\\dp_eval"
    )
    ############################# Output path for the results ############################
    parser.add_argument(
        "-o_acc_val",
        "--result_dir_acc_val",
        type=str,
        default="results_e2e\\val_acc",
    )
    parser.add_argument(
        "-r_td",
        "--result_dir_td",
        type=str,
        default="results_e2e\\elapsed_time",
    )
    parser.add_argument(
        "-o_acc_val_adv_dp",
        "--result_dir_acc_val_adv_dp",
        type=str,
        default="results_e2e\\val_acc_adv_dp",
    )
    parser.add_argument(
        "-o_pv_spent",
        "--result_dir_privacy_spent",
        type=str,
        default="results_e2e\\pv_spent",
    )

    args = parser.parse_args()
    return args
