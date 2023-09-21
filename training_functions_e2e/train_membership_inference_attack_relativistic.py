import torch


from torch.utils.data import DataLoader
from utils_e2e import cor_seq_counter_list, save_model_ckpt_cls


from preprocess_data import (
    ClassificationDatasetGdrSpkr,
    collateGdrSpkr,
    create_dataset_arguments_bkt,
)


def train_memb_inf_attk_relativistic(
    args,
    hparams,
    device,
    model_collection,
    opt,
    loss_props,
    attk_props,
    status,
    ckpt_dvec_cls_shadow,
    checkpoint_memb_attk_cls,
    utts_counts_max,
    filenames_and_dirs,
):
    """Train the relativistic membership inference attacker.

    Args:
        args: The required arguments to be parsed within the function.
        hparams (HyperParams): The parameters from the dataclass.
        device: The device to run the simulations on.
        model_collection: The collection of models.
        opt: The optimizer type.
        loss_props (dict): The dictionary of loss props.
        attk_props(dict): The dictionary of attack props.
        status (string): The status of the trained shadow model checkpoints.
        loss: The loss.
        ckpt_dvec_cls_shadow: The checkpoints of the shadow classifier.
        utts_counts_max: The max utts counter.
        filenames_and_dirs: file dir for saving the checkpoints.
    """

    # Create datasets: train-test-eval
    data_dir_train, speaker_infos_train = create_dataset_arguments_bkt(
        args,
        hparams.buckets,
        args.dp_data_dir_train,
        utts_counts_max,
    )
    dataset_train = ClassificationDatasetGdrSpkr(
        data_dir_train,
        speaker_infos_train,
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

    # Shadow classifier previously trained on public data
    shadow_dvec_cls_e2e = model_collection["shadow_model"](args).to(device)

    if ckpt_dvec_cls_shadow is not None:
        ckpt_dvec_cls_shadow = torch.load(ckpt_dvec_cls_shadow)

        if status == "dp":
            pretrained_dict = {
                k.removeprefix("_module."): v
                for k, v in ckpt_dvec_cls_shadow[hparams.model_str].items()
            }

            # Load the corresponding parameters of the layers
            model_dict = shadow_dvec_cls_e2e.state_dict()
            model_dict.update(pretrained_dict)
            shadow_dvec_cls_e2e.load_state_dict(model_dict)
        else:
            shadow_dvec_cls_e2e.load_state_dict(ckpt_dvec_cls_shadow[hparams.model_str])

    # Set the gradients of the parameters to False for shadow model
    for _, qs in shadow_dvec_cls_e2e.named_parameters():
        qs.requires_grad = False

    # Create membership inference attack model and optimizer
    memb_attk_cls = model_collection["memb_attk_model"](args).to(device)
    opt_memb_attk_cls = opt["opt"](memb_attk_cls.parameters(), lr=hparams.lr_memb_cls)

    # Load available checkpoints for the membership inference attack
    if checkpoint_memb_attk_cls is not None:
        checkpoint_memb_attk_cls = torch.load(checkpoint_memb_attk_cls)
        memb_attk_cls.load_state_dict(checkpoint_memb_attk_cls["model"])
        opt_memb_attk_cls.load_state_dict(checkpoint_memb_attk_cls["optimizer"])

        start_epoch_available = checkpoint_memb_attk_cls.get(hparams.start_epoch)

        if start_epoch_available:
            start_epoch = start_epoch_available + 1
        else:
            start_epoch = 0
    else:
        start_epoch = 0

    # The loss to train the membership inference attack
    ce_loss = loss_props["loss"]().to(device)
    # softmax = torch.nn.Softmax(dim=1)

    # Create attack
    attk = attk_props["name"](
        shadow_dvec_cls_e2e,
        device,
        attk_props["tgt_label"],
        attk_props["cost"],
        hparams,
    )

    # The membership inference attacker
    for epoch in range(start_epoch, start_epoch + args.epoch):
        # Training samples
        public_data_loader = DataLoader(
            dataset_train,
            batch_size=len(output_composite_flattened),
            collate_fn=collateGdrSpkr,
            drop_last=True,
        )

        ########################################################################
        ##########                  Create buffer samples                 ######
        ########################################################################

        # Public samples
        x_public, gdr_idx_public, spk_idx_public = next(iter(public_data_loader))

        x_public = x_public.reshape(-1, args.seg_len, args.feature_dim).to(device)
        gdr_idx_public = gdr_idx_public.view(-1).to(device)
        spk_idx_public = spk_idx_public.view(-1).to(device)

        # Create normal and adversarial latent features for the public dataset
        x_adv_public, _ = attk(x_public, spk_idx_public)

        _, _, emb_pub, _ = shadow_dvec_cls_e2e(x_public)
        _, _, emb_pub_adv, _ = shadow_dvec_cls_e2e(x_adv_public)

        valid = (
            torch.ones([x_public.shape[0]], dtype=torch.float32).view(-1, 1).to(device)
        )
        fake = (
            torch.zeros([x_adv_public.shape[0]], dtype=torch.float32)
            .view(-1, 1)
            .to(device)
        )

        memb_attk_cls.train()

        real_feats, _ = memb_attk_cls(emb_pub)
        fake_feats, _ = memb_attk_cls(emb_pub_adv)

        real_loss = ce_loss(real_feats - fake_feats.mean(0, keepdim=True), valid)
        fake_loss = ce_loss(fake_feats - real_feats.mean(0, keepdim=True), fake)

        loss = (real_loss + fake_loss) / 2

        opt_memb_attk_cls.zero_grad()
        loss.backward()
        opt_memb_attk_cls.step()

        # Save the checkpoint for "model"
        if epoch % args.save_every == 0:
            memb_attk_cls.to("cpu")

            save_model_ckpt_cls(
                epoch,
                -1,
                memb_attk_cls,
                None,
                opt_memb_attk_cls,
                ce_loss,
                loss,
                None,
                filenames_and_dirs["filename_dir"],
            )

            memb_attk_cls.to(device)

        # Evaluation
        with torch.no_grad():
            real_loss = ce_loss(real_feats - fake_feats.mean(0, keepdim=True), valid)
            fake_loss = ce_loss(fake_feats - real_feats.mean(0, keepdim=True), fake)

            loss = (real_loss + fake_loss) / 2

            print(f"Epoch: {epoch}, Loss: {loss:.3f}")
