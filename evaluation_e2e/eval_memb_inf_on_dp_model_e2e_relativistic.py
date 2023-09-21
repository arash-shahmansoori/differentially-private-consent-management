import torch


from torch.utils.data import DataLoader
from utils_e2e import cor_seq_counter_list
from preprocess_data import (
    ClassificationDatasetGdrSpkr,
    collateGdrSpkr,
    create_dataset_arguments_bkt,
)


def eval_memb_inf_performance_on_dp_model_e2e_relativistic(
    args,
    hparams,
    device,
    model_collection,
    loss_props,
    attk_props,
    ckpt_dvec_cls_shadow,
    ckpt_dvec_cls,
    utts_counts_max,
    ckpt_memb_attk_cls,
):
    """Evaluate the performance of the relativistic membership inference attack on
    differentially private model.

    Args:
        args: The required arguments to be parsed within the function.
        hparams (HyperParams): The parameters from the dataclass.
        device: The device to run the simulations on.
        model_collection: The collection of models.
        loss_props (dict): The dictionary of loss props.
        attk_props (dict): The dictionary of attack properties.
        ckpt_dvec_cls_shadow: The available shadow model checkpoints.
        ckpt_dvec_cls: The available checkpoints.
        utts_counts_max: The max utts counter.
        ckpt_memb_attk_cls: The checkpoints of the attacker model.
    """

    # Create dataset
    data_dir, speaker_infos = create_dataset_arguments_bkt(
        args,
        hparams.buckets,
        args.dp_data_dir_train,
        utts_counts_max,
    )
    dataset_train = ClassificationDatasetGdrSpkr(
        data_dir,
        speaker_infos,
        args.n_utterances_labeled,
        args.seg_len,
    )

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

    # Create buckets of speakers
    labels = [i for i in range(args.n_speakers)]
    outputs = cor_seq_counter_list(
        len(labels),
        args.spk_per_bucket,
        args.spk_per_bucket,
    )
    output_composite = [outputs[i] for i in hparams.buckets]
    output_composite_flattened = [i for s in output_composite for i in s]

    # Create model
    model = model_collection["model"](args).to(device)

    # Load available checkpoints
    if ckpt_dvec_cls is not None:
        ckpt_dvec_cls = torch.load(ckpt_dvec_cls)

        pretrained_dict = {
            k.removeprefix("_module."): v
            for k, v in ckpt_dvec_cls[hparams.model_str].items()
        }

        # Load the corresponding parameters of the layers
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    # Create shadow model
    shadow_model = model_collection["shadow_model"](args).to(device)

    # Load available checkpoints
    if ckpt_dvec_cls_shadow is not None:
        ckpt_dvec_cls_shadow = torch.load(ckpt_dvec_cls_shadow)

        pretrained_dict_shadow = {
            k.removeprefix("_module."): v
            for k, v in ckpt_dvec_cls_shadow[hparams.model_str].items()
        }

        # Load the corresponding parameters of the layers
        shadow_model_dict = shadow_model.state_dict()
        shadow_model_dict.update(pretrained_dict_shadow)
        shadow_model.load_state_dict(shadow_model_dict)

    # Set the gradients of the parameters to False for shadow model
    for _, qs in shadow_model.named_parameters():
        qs.requires_grad = False

    # Create membership inference attack model
    memb_attk_cls = model_collection["memb_attk_model"](args).to(device)
    ce_loss = loss_props["loss"]().to(device)

    # Load available checkpoints for the membership inference attack
    if ckpt_memb_attk_cls is not None:
        ckpt_memb_attk_cls = torch.load(ckpt_memb_attk_cls)
        memb_attk_cls.load_state_dict(ckpt_memb_attk_cls["model"])
        memb_attk_cls.eval()

    # Create attack
    attk = attk_props["name"](
        shadow_model,
        device,
        attk_props["tgt_label"],
        attk_props["cost"],
        hparams,
    )

    # The main differentially private training loop
    for epoch in range(args.epoch_test):
        # Create public samples
        public_data_loader = DataLoader(
            dataset_train,
            batch_size=len(output_composite_flattened),
            collate_fn=collateGdrSpkr,
            drop_last=True,
        )

        x_public, gdr_idx_public, spk_idx_public = next(iter(public_data_loader))

        x_public = x_public.reshape(-1, args.seg_len, args.feature_dim).to(device)
        gdr_idx_public = gdr_idx_public.view(-1).to(device)
        spk_idx_public = spk_idx_public.to(device)

        # Create private samples
        private_data_loader = DataLoader(
            dataset_test,
            batch_size=len(output_composite_flattened),
            collate_fn=collateGdrSpkr,
            drop_last=True,
        )

        x_private, gdr_idx_private, spk_idx_private = next(iter(private_data_loader))

        x_private = x_private.reshape(-1, args.seg_len, args.feature_dim).to(device)
        gdr_idx_private = gdr_idx_private.view(-1).to(device)
        spk_idx_private = spk_idx_private.to(device)

        # Create normal and adversarial features
        x_public_adv, _ = attk(x_public, spk_idx_public)
        x_private_adv, _ = attk(x_private, spk_idx_private)

        _, _, emb_pub, _ = model(x_public)
        _, _, emb_pub_adv, _ = model(x_public_adv)

        real_feats_pub, _ = memb_attk_cls(emb_pub)
        fake_feats_pub, _ = memb_attk_cls(emb_pub_adv)

        _, _, emb_pvt, _ = model(x_private)
        _, _, emb_pvt_adv, _ = model(x_private_adv)

        real_feats_pvt, _ = memb_attk_cls(emb_pvt)
        fake_feats_pvt, _ = memb_attk_cls(emb_pvt_adv)

        valid_public = (
            torch.ones([x_public.shape[0]], dtype=torch.float32).view(-1, 1).to(device)
        )
        fake_public = (
            torch.zeros([x_public_adv.shape[0]], dtype=torch.float32)
            .view(-1, 1)
            .to(device)
        )

        valid_private = (
            torch.ones([x_private.shape[0]], dtype=torch.float32).view(-1, 1).to(device)
        )
        fake_private = (
            torch.zeros([x_private_adv.shape[0]], dtype=torch.float32)
            .view(-1, 1)
            .to(device)
        )

        # Evaluation
        with torch.no_grad():
            real_loss_pub = ce_loss(
                real_feats_pub - fake_feats_pub.mean(0, keepdim=True),
                valid_public,
            )
            fake_loss_pub = ce_loss(
                fake_feats_pub - real_feats_pub.mean(0, keepdim=True),
                fake_public,
            )

            real_loss_pvt = ce_loss(
                real_feats_pvt - fake_feats_pub.mean(0, keepdim=True),
                valid_private,
            )
            fake_loss_pvt = ce_loss(
                fake_feats_pvt - real_feats_pub.mean(0, keepdim=True),
                fake_private,
            )

            loss_pub = (real_loss_pub + fake_loss_pub) / 2
            loss_pvt = (real_loss_pvt + fake_loss_pvt) / 2

            print(f"Epoch: {epoch}, LossPub: {loss_pub:.3f}, LossPvt: {loss_pvt:.3f}")
