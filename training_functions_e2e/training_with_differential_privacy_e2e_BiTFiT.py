import torch
import torch.nn as nn

from torch.utils.data import DataLoader


from utils_e2e import (
    create_filenames_results_adv_dp,
    cor_seq_counter_list,
    get_logger,
    dataset_kwargs,
    model_kwargs,
    opt_kwargs,
    loss_kwargs,
    save_as_json,
)
from agent_supervised_e2e import AgentSupervisedE2E

from preprocess_data import SubDatasetGdrSpk, collateGdrSpkr
from evaluation_e2e import eval_dp_per_epoch_supervised_composite_e2e
from early_stop import EarlyStoppingCustomComposite

from fastDP import PrivacyEngine
from opacus.validators import ModuleValidator

from .train_dp_epoch_supervised_selective_comp_e2e import (
    train_dp_per_epoch_supervised_selective_composite_e2e,
)


def train_with_dp_e2e_BiTFiT(
    args,
    hparams,
    device,
    sub_sample,
    model_collection,
    opt,
    privacy_props,
    attk_props,
    ckpt_dvec_cls_shadow,
    ckpt_dvec_cls,
    status_dp_dvec_cls,
    dp_mode,
    utts_counts_max,
    filenames_dvecs_cls_and_dirs_dp,
):
    """Differantially private training with bias term fine-tuning.

    Args:
        args: The required arguments to be parsed within the function.
        hparams (HyperParams): The parameters from the dataclass.
        device: The device to run the simulations on.
        sub_sample: Sub_sample object for improved privacy.
        model_collection: The collection of models.
        opt: The optimizer type.
        privacy_props (dict): The dictionary of privacy properties.
        attk_props (dict): The dictionary of attack properties.
        ckpt_dvec_cls_shadow: The checkpoints of the shadow classifier.
        ckpt_dvec_cls: The checkpoints of the dp classifier.
        status_dp_dvec_cls: status for the dp classifier.
        dp_mode: The mode of dp training.
        utts_counts_max: The max utts counter.
        filenames_dvecs_cls_and_dirs_dp: file dir for saving the checkpoints.
    """

    # Create paths and filenames for saving the metrics
    paths_filenames = create_filenames_results_adv_dp(
        args,
        dp_mode,
        hparams.buckets_dp,
        utts_counts_max,
        privacy_props["eps"],
        privacy_props["delta"],
        args.agnt_num,
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
    dvec_cls_e2e = model_collection["model"](args)
    dvec_cls_e2e = ModuleValidator.fix(dvec_cls_e2e).to(device)

    optimizer = opt["opt"](dvec_cls_e2e.parameters(), lr=hparams.lr_cls, amsgrad=True)

    # CrossEntropy loss
    ce_loss = nn.CrossEntropyLoss().to(device)

    # # Load available checkpoints
    # if ckpt_dvec_cls is not None:
    #     ckpt_dvec_cls = torch.load(ckpt_dvec_cls)

    #     dvec_cls_e2e.load_state_dict(ckpt_dvec_cls[hparams.model_str])
    #     optimizer.load_state_dict(ckpt_dvec_cls[hparams.opt_str])

    #     start_epoch_available = ckpt_dvec_cls.get(hparams.start_epoch)

    #     if start_epoch_available and status_dp_dvec_cls == "dp":
    #         start_epoch = start_epoch_available + 1
    #     else:
    #         start_epoch = 0
    # else:
    #     start_epoch = 0
    start_epoch = 0

    for name, param in dvec_cls_e2e.named_parameters():
        if ".bias" not in name:
            param.requires_grad_(False)

    # Privacy engine
    # dvec_cls_e2e, optimizer = privacy_props["privacy_engine"].custom_make_private(
    #     module=dvec_cls_e2e,
    #     batch_size=privacy_props["bs"],
    #     sample_size=len(privacy_props["dataset"]),
    #     optimizer=optimizer,
    #     noise_multiplier=privacy_props["sigma"],
    #     max_grad_norm=hparams.clipping_norm,
    # )

    # Privacy engine
    privacy_engine = PrivacyEngine(
        dvec_cls_e2e,
        batch_size=privacy_props["bs"],
        sample_size=len(privacy_props["dataset"]),
        noise_multiplier=privacy_props["sigma"],
        epochs=privacy_props["num_iter"],
        clipping_mode=privacy_props["clipping_mode"],
        origin_params=args.origin_params,
    )
    privacy_engine.attach(optimizer)

    # Instantiate the Agent class
    agent = AgentSupervisedE2E(args, device, hparams)

    # # Shadow classifier previously trained on public data
    # shadow_dvec_cls_e2e = model_collection["shadow_model"](args).to(device)

    # if ckpt_dvec_cls_shadow is not None:
    #     ckpt_dvec_cls_shadow = torch.load(ckpt_dvec_cls_shadow)

    #     shadow_dvec_cls_e2e.load_state_dict(ckpt_dvec_cls_shadow[hparams.model_str])

    # # Set the gradients of the parameters to False for shadow model
    # for _, qs in shadow_dvec_cls_e2e.named_parameters():
    #     qs.requires_grad = False

    # # Create attack
    # attk = attk_props["name"](
    #     shadow_dvec_cls_e2e,
    #     device,
    #     attk_props["tgt_label"],
    #     attk_props["cost"],
    #     hparams,
    # )

    # Initializing early stoppings for the buckets
    if args.early_stopping:
        early_stopping = EarlyStoppingCustomComposite(args, mode="dp")

    # Create kwargs for the training/validation function
    kwargs_dataset = dataset_kwargs(
        SubDatasetGdrSpk,
        collateGdrSpkr,
        None,
        privacy_props["dataset"],
    )

    kwargs_model = model_kwargs(agent, None, dvec_cls_e2e)
    kwargs_filename_dvec_cls = filenames_dvecs_cls_and_dirs_dp

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

    # Initialize validation accuracy and privacy spent
    val_acc = []
    pv_spent_list = []

    # Initialize the elapsed time per epoch
    td_per_epoch = []

    # The main differentially private training loop
    for epoch in range(start_epoch, start_epoch + privacy_props["num_iter"]):
        # Create private samples
        private_data_loader = DataLoader(
            privacy_props["dataset"],
            batch_size=len(output_composite_flattened),
            collate_fn=collateGdrSpkr,
            drop_last=True,
        )

        x_private, _, spk_idx_private = next(iter(private_data_loader))

        x_private = x_private.reshape(-1, args.seg_len, args.feature_dim).to(device)
        spk_idx_private = spk_idx_private.to(device)

        # Sub_sample
        indx_comp = sub_sample.utt_index_per_bucket_collection(
            [len(output_composite_flattened)],
            privacy_props["utts_per_spk"],
        )

        # x_sub_private = x_private[indx_comp]
        # spk_sub_idx_private = spk_idx_private[indx_comp]
        # # Create normal and adversarial features
        # x_sub_private_adv, _ = attk(x_sub_private, spk_sub_idx_private)
        # x_pvt_comb = torch.cat([x_sub_private, x_sub_private_adv], dim=0).reshape(
        #     -1,
        #     args.seg_len,
        #     args.feature_dim,
        # )
        x_pvt_comb = x_private[indx_comp]

        # spk_pvt_com = torch.cat([spk_sub_idx_private, spk_sub_idx_private]).view(-1)
        spk_pvt_com = spk_idx_private[indx_comp]

        input_data = {"x": x_pvt_comb, "y": spk_pvt_com}

        # Train the d-vectors per epoch and evaluate the performance
        td, train_out = train_dp_per_epoch_supervised_selective_composite_e2e(
            input_data,
            logger,
            epoch,
            **kwargs_training_val,
        )

        # Store the elapsed time per epoch in a list
        td_per_epoch.append(td)

        # Evaluate the performance per epoch
        val_out = eval_dp_per_epoch_supervised_composite_e2e(
            args,
            device,
            output_composite_flattened,
            val_acc,
            epoch,
            **kwargs_training_val,
        )

        # Update early stopping parameters for the buckets
        if args.early_stopping:
            early_stopping(torch.tensor(val_out["val_acc"]).view(-1)[-1], epoch)

        # Get the privacy spent
        epsilon = privacy_props["compute_eps_ews"](
            hparams.sigma,
            privacy_props["points"],
            privacy_props["bs"] / (privacy_props["len_dataset"]),
            hparams.delta,
            order=2,
        )
        eps = epsilon[0][0]
        pv_spent_list.append(eps)

        # Break training if the early stopping status is ``True''
        # after completion of progressive registrations
        if train_out["early_stops_status"][-1]:
            epsilon = privacy_props["compute_eps_ews"](
                hparams.sigma,
                [epoch],
                privacy_props["bs"] / (privacy_props["len_dataset"]),
                hparams.delta,
                order=2,
            )
            eps = epsilon[0][0]
            print(f"(ε = {eps}, δ = {hparams.delta})")

            pv_spent_list.append(eps)

            # save_as_json(
            #     paths_filenames["dir_privacy_spent"],
            #     paths_filenames["filename_privacy_spent"],
            #     pv_spent_list,
            # )

            # # Save the required validation metrics as JSON files
            # save_as_json(
            #     paths_filenames["dir_acc_val_adv_dp"],
            #     paths_filenames["filename_acc_val_adv_dp"],
            #     val_out["val_acc"],
            # )

            break

        # else:
        # Save the required validation metrics as JSON files
        # save_as_json(
        #     paths_filenames["dir_acc_val_adv_dp"],
        #     paths_filenames["filename_acc_val_adv_dp"],
        #     val_out["val_acc"],
        # )
        # save_as_json(
        #     paths_filenames["dir_privacy_spent"],
        #     paths_filenames["filename_privacy_spent"],
        #     pv_spent_list,
        # )
