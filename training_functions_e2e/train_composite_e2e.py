import torch
import torch.nn as nn


from utils_e2e import (
    cor_seq_counter_list,
    get_logger,
    dataset_kwargs,
    model_kwargs,
    opt_kwargs,
    loss_kwargs,
)


from .train_epoch_supervised_selective_composite_e2e import (
    train_per_epoch_supervised_selective_composite_e2e,
)
from evaluation_e2e import eval_per_epoch_supervised_composite_e2e
from early_stop import EarlyStoppingCustomComposite

from preprocess_data import (
    ClassificationDatasetGdrSpkr,
    SubDatasetGdrSpk,
    collateGdrSpkr,
    create_dataset_arguments_bkt,
)


from agent_supervised_e2e import AgentSupervisedE2E


def train_comp_e2e(
    args,
    hparams,
    device,
    model_collection,
    opt,
    ckpt_dvec_cls,
    utts_counts_max,
    filenames_dvecs_cls_and_dirs,
):
    # Create the index list of speakers
    labels = [i for i in range(args.n_speakers)]
    outputs = cor_seq_counter_list(
        len(labels),
        args.spk_per_bucket,
        args.spk_per_bucket,
    )
    output_composite = [outputs[i] for i in hparams.buckets]
    output_composite_flattened = [i for s in output_composite for i in s]

    # Create training/validation datasets
    data_dir_train, speaker_infos_train = create_dataset_arguments_bkt(
        args,
        hparams.buckets,
        args.dp_data_dir_train,
        utts_counts_max,
    )

    # Uncomment if a portion of samples of a bucket are private
    data_dir_test, speaker_infos_test = create_dataset_arguments_bkt(
        args,
        hparams.buckets,
        args.dp_data_dir_test,
        utts_counts_max,
    )

    dataset = ClassificationDatasetGdrSpkr(
        data_dir_train,
        speaker_infos_train,
        args.n_utterances_labeled,
        args.seg_len,
    )

    dataset_test = ClassificationDatasetGdrSpkr(
        data_dir_test,
        speaker_infos_test,
        args.n_utterances_labeled,
        args.seg_len,
    )

    # Classifier to be trained on the contrastive embedding replay
    dvec_cls_e2e = model_collection["model"](args).to(device)
    optimizer = opt["opt"](dvec_cls_e2e.parameters(), lr=hparams.lr_cls, amsgrad=True)

    # # Load available checkpoints for the speaker recognition in latent space
    # if ckpt_dvec_cls is not None:
    #     ckpt_dvec_cls = torch.load(ckpt_dvec_cls)
    #     dvec_cls_e2e.load_state_dict(ckpt_dvec_cls[hparams.model_str])
    #     optimizer.load_state_dict(ckpt_dvec_cls[hparams.opt_str])

    #     start_epoch_available = ckpt_dvec_cls.get(hparams.start_epoch)

    #     if start_epoch_available:
    #         start_epoch = start_epoch_available + 1
    #     else:
    #         start_epoch = 0
    # else:
    #     start_epoch = 0
    start_epoch = 0

    # Initializing early stoppings for the buckets
    if args.early_stopping:
        early_stopping = EarlyStoppingCustomComposite(args)

    # The loss
    ce_loss = nn.CrossEntropyLoss().to(device)  # CrossEntropy loss for the classifier

    # Instantiate the Agent class
    agent = AgentSupervisedE2E(args, device, hparams)

    # Create kwargs for the training/validation function
    kwargs_dataset = dataset_kwargs(
        SubDatasetGdrSpk,
        collateGdrSpkr,
        dataset,
        dataset_test,
    )
    kwargs_model = model_kwargs(agent, None, dvec_cls_e2e)
    kwargs_filenames_dvecs_cls = (
        filenames_dvecs_cls_and_dirs
        if filenames_dvecs_cls_and_dirs != None
        else {"filename_dir": None}
    )
    kwargs_opt = opt_kwargs(None, None, opt["opt"], optimizer, early_stopping)
    kwargs_loss = loss_kwargs(None, ce_loss)

    # Combine training and validation kwargs
    kwargs_training_val = (
        kwargs_dataset
        | kwargs_model
        | kwargs_opt
        | kwargs_loss
        | kwargs_filenames_dvecs_cls
    )

    # Logging
    logger = get_logger()

    # Initialize validation accuracy
    val_acc, train_acc = [], []

    # Initialize the elapsed time per epoch
    td_per_epoch = []

    for epoch in range(start_epoch, start_epoch + args.epoch):
        # Train the d-vectors per epoch and evaluate the performance
        td, train_out = train_per_epoch_supervised_selective_composite_e2e(
            args,
            device,
            output_composite_flattened,
            epoch,
            logger,
            **kwargs_training_val,
        )

        # Store the elapsed time per epoch in a list
        td_per_epoch.append(td)

        # Evaluate the performance per epoch
        val_out = eval_per_epoch_supervised_composite_e2e(
            args,
            device,
            output_composite_flattened,
            val_acc,
            train_acc,
            epoch,
            **kwargs_training_val,
        )

        # Update early stopping parameters for the buckets
        if args.early_stopping:
            early_stopping(torch.tensor(val_out["train_acc"]).view(-1)[-1], epoch)

        # Break training if the early stopping status is ``True''
        # after completion of progressive registrations
        if train_out["early_stops_status"][-1]:
            break
