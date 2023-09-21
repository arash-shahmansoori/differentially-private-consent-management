import torch

from torch.utils.data import DataLoader


def eval_adv_per_epoch_supervised_composite_e2e(
    args,
    device,
    output_composite,
    val_acc,
    train_acc,
    epoch,
    attk,
    **kwargs_validation,
):
    sub_lbs_current = output_composite

    loader_current = DataLoader(
        kwargs_validation["dataset"],
        batch_size=len(sub_lbs_current),
        collate_fn=kwargs_validation["collateGdrSpkr"],
        drop_last=True,
    )

    mel_db_batch = next(iter(loader_current))

    x, _, spk = mel_db_batch
    x = x.reshape(-1, args.seg_len, args.feature_dim)
    x, spk = x.to(device), spk.to(device)

    validation_loader_current = DataLoader(
        kwargs_validation["dataset_test"],
        batch_size=len(sub_lbs_current),
        collate_fn=kwargs_validation["collateGdrSpkr"],
        drop_last=True,
    )

    mel_db_batch_validation = next(iter(validation_loader_current))

    x_val, _, spk_val = mel_db_batch_validation
    x_val = x_val.reshape(-1, args.seg_len, args.feature_dim)
    x_val, spk_val = x_val.to(device), spk_val.to(device)

    # Create normal and adversarial latent features
    x_adv, _ = attk(x, spk)
    x_val_adv, _ = attk(x_val, spk_val)

    x_adv_public = torch.cat([x, x_adv], dim=0).reshape(
        -1,
        args.seg_len,
        args.feature_dim,
    )
    labels = torch.cat([spk, spk]).view(-1)

    x_val_adv_private = torch.cat([x_val, x_val_adv], dim=0).reshape(
        -1,
        args.seg_len,
        args.feature_dim,
    )
    labels_val = torch.cat([spk_val, spk_val]).view(-1)

    # Compute performance measures of the classifier
    acc, loss = kwargs_validation["agent"].accuracy_loss(
        kwargs_validation["classifier"],
        kwargs_validation["ce_loss"],
        x_adv_public,
        labels,
    )

    acc_val, loss_val = kwargs_validation["agent"].accuracy_loss(
        kwargs_validation["classifier"],
        kwargs_validation["ce_loss"],
        x_val_adv_private,
        labels_val,
    )

    if args.log_training:
        loss_val_display = loss_val.item()
        acc_val_display = acc_val.item()

        loss_display = loss.item()
        acc_display = acc.item()

        epoch_display = f"Train Epoch: {epoch}| "

        val_loss_display = f"ValLoss:{loss_val_display:0.3f}| "
        val_acc_display = f"ValAcc:{acc_val_display:0.3f}| "

        display_loss = f"TrainLoss:{loss_display:0.3f}| "
        display_acc = f"TrainAcc:{acc_display:0.3f}| "

        print(
            epoch_display,
            val_loss_display,
            val_acc_display,
            display_loss,
            display_acc,
        )

    val_acc.append(acc_val.item())
    train_acc.append(acc.item())

    out = {"val_acc": val_acc, "train_acc": train_acc}

    return out
