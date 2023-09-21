import torch

from torch.utils.data import DataLoader


def eval_dp_per_epoch_supervised_composite_e2e(
    args,
    device,
    output_composite,
    val_acc,
    epoch,
    **kwargs_validation,
):
    sub_lbs_current = output_composite

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

    # Create normal and adversarial latent features for the public dataset
    # x_val_adv, _ = attk(x_val, spk_val)

    # x_val_adv_public = torch.cat([x_val, x_val_adv], dim=0).reshape(
    #     -1,
    #     args.seg_len,
    #     args.feature_dim,
    # )
    # labels_val = torch.cat([spk_val, spk_val]).view(-1)

    x_val_adv_public = x_val
    labels_val = spk_val

    # Compute performance measures of the classifier
    acc_val, loss_val = kwargs_validation["agent"].accuracy_loss(
        kwargs_validation["classifier"],
        kwargs_validation["ce_loss"],
        x_val_adv_public,
        labels_val,
    )

    if args.log_training:
        loss_val_display = loss_val.item()
        acc_val_display = acc_val.item()

        epoch_display = f"Train Epoch: {epoch}| "

        val_loss_display = f"ValLoss:{loss_val_display:0.3f}| "
        val_acc_display = f"ValAcc:{acc_val_display:0.3f}| "

        print(epoch_display, val_loss_display, val_acc_display)

    val_acc.append(acc_val.item())

    out = {"val_acc": val_acc}

    return out
