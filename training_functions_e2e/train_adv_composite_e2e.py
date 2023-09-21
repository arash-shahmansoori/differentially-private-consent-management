import torch
import torch.nn as nn


from torch.utils.data import DataLoader
from utils_e2e import (
    dataset_kwargs,
    model_kwargs,
    opt_kwargs,
    loss_kwargs,
    get_logger,
    cor_seq_counter_list,
)
from evaluation_e2e import eval_adv_per_epoch_supervised_composite_e2e
from early_stop import EarlyStoppingCustomComposite

from agent_supervised_e2e import AgentSupervisedE2E
from preprocess_data import (
    ClassificationDatasetGdrSpkr,
    SubDatasetGdrSpk,
    collateGdrSpkr,
    create_dataset_arguments_bkt,
)
from .train_adv_epoch_supervised_selective_comp_e2e import (
    train_adv_per_epoch_supervised_selective_composite_e2e,
)


def train_adv_comp_e2e(
    args,
    hparams,
    device,
    model_collection,
    opt,
    attk_props,
    ckpt_dvec_cls_shadow,
    ckpt_dvec_cls,
    status_cls,
    utts_counts_max,
    filenames_dvecs_cls_and_dirs,
):
    """Adversarial traning with public data.

    Args:
        args: The required arguments to be parsed within the function.
        hparams (HyperParams): The parameters from the dataclass.
        device: The device to run the simulations on.
        model_collection: The collection of models.
        opt: The optimizer type.
        attk_props(dict): The dictionary of attack props.
        ckpt_dvec_cls_shadow: The checkpoints of the shadow classifier.
        ckpt_dvec_cls: The checkpoints of the classifier.
        status_cls: The status of the classifier.
        utts_counts_max: The max utts counter.
        filenames_dvecs_cls_and_dirs: file dir for saving the checkpoints.
    """

    # Create datasets: train-test-eval
    data_dir_train, speaker_infos_train = create_dataset_arguments_bkt(
        args,
        hparams.buckets,
        args.dp_data_dir_train,
        utts_counts_max,
    )
    dataset = ClassificationDatasetGdrSpkr(
        data_dir_train,
        speaker_infos_train,
        args.n_utterances_labeled,
        args.seg_len,
    )

    # Uncomment for parial private data of a given bucket
    data_dir_test, speaker_infos_test = create_dataset_arguments_bkt(
        args,
        hparams.buckets,
        args.dp_data_dir_test,
        utts_counts_max,
    )
    dataset_test = ClassificationDatasetGdrSpkr(
        data_dir_test,
        speaker_infos_test,
        args.n_utterances_labeled,
        args.seg_len,
    )

    # Create buckets of speakers
    labels = [i for i in range(args.n_speakers)]
    outputs = cor_seq_counter_list(
        len(labels),
        args.spk_per_bucket,
        args.spk_per_bucket,
    )
    output_composite = [outputs[i] for i in hparams.buckets]
    output_composite_flattened = [i for s in output_composite for i in s]

    # Create model and optimizer
    dvec_cls_e2e = model_collection["model"](args).to(device)
    optimizer = opt["opt"](dvec_cls_e2e.parameters(), lr=hparams.lr_cls, amsgrad=True)

    # Load available checkpoints
    if ckpt_dvec_cls is not None:
        ckpt_dvec_cls = torch.load(ckpt_dvec_cls)

        dvec_cls_e2e.load_state_dict(ckpt_dvec_cls[hparams.model_str])
        optimizer.load_state_dict(ckpt_dvec_cls[hparams.opt_str])

        start_epoch_available = ckpt_dvec_cls.get(hparams.start_epoch)

        if start_epoch_available and status_cls != "shadow":
            start_epoch = start_epoch_available + 1
        else:
            start_epoch = 0
    else:
        start_epoch = 0

    # Shadow classifier previously trained on public data
    shadow_dvec_cls_e2e = model_collection["shadow_model"](args).to(device)

    if ckpt_dvec_cls_shadow is not None:
        ckpt_dvec_cls_shadow = torch.load(ckpt_dvec_cls_shadow)

        shadow_dvec_cls_e2e.load_state_dict(ckpt_dvec_cls_shadow[hparams.model_str])

    # Set the gradients of the parameters to False for shadow model
    for _, qs in shadow_dvec_cls_e2e.named_parameters():
        qs.requires_grad = False

    # Initializing early stoppings for the buckets
    if args.early_stopping:
        early_stopping = EarlyStoppingCustomComposite(args)

    ce_loss = nn.CrossEntropyLoss().to(device)  # CrossEntropy loss

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
    kwargs_filename_dvec_cls = filenames_dvecs_cls_and_dirs

    kwargs_opt = opt_kwargs(
        None,
        None,
        opt["opt"],
        optimizer,
        early_stopping,
    )

    kwargs_loss = loss_kwargs(None, ce_loss)

    # Combine training and validation kwargs
    kwargs_training_val = (
        kwargs_dataset
        | kwargs_model
        | kwargs_opt
        | kwargs_loss
        | kwargs_filename_dvec_cls
    )

    # Logging
    logger = get_logger()

    # Initialize validation accuracy
    val_acc, train_acc = [], []

    # Create attack
    attk = attk_props["name"](
        shadow_dvec_cls_e2e,
        device,
        attk_props["tgt_label"],
        attk_props["cost"],
        hparams,
    )

    # Initialize the elapsed time per epoch
    td_per_epoch = []

    # The membership inference attacker
    for epoch in range(start_epoch, start_epoch + args.epoch):
        train_sub_loader = DataLoader(
            dataset,
            batch_size=len(output_composite_flattened),
            shuffle=False,
            collate_fn=collateGdrSpkr,
            drop_last=True,
        )

        mel_db_batch = next(iter(train_sub_loader))

        x, _, spk = mel_db_batch
        x = x.reshape(-1, args.seg_len, args.feature_dim).to(device)
        spk = spk.to(device)

        # Create normal and adversarial latent features for the public dataset
        x_adv, _ = attk(x, spk)

        x_adv_public = torch.cat([x, x_adv], dim=0).reshape(
            -1,
            args.seg_len,
            args.feature_dim,
        )
        labels = torch.cat([spk, spk]).view(-1)

        input_data = {"x": x_adv_public, "y": labels}

        # Train the d-vectors per epoch and evaluate the performance
        td, train_out = train_adv_per_epoch_supervised_selective_composite_e2e(
            input_data,
            logger,
            epoch,
            **kwargs_training_val,
        )

        # Store the elapsed time per epoch in a list
        td_per_epoch.append(td)

        # Evaluate the performance per epoch
        val_out = eval_adv_per_epoch_supervised_composite_e2e(
            args,
            device,
            output_composite_flattened,
            val_acc,
            train_acc,
            epoch,
            attk,
            **kwargs_training_val,
        )

        # Update early stopping parameters for the buckets
        if args.early_stopping:
            early_stopping(torch.tensor(val_out["train_acc"]).view(-1)[-1], epoch)

        # Break training if the early stopping status is ``True''
        # after completion of progressive registrations
        if train_out["early_stops_status"][-1]:
            break
