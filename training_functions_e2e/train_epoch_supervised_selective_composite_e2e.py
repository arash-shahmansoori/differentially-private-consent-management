from torch.utils.data import DataLoader
from utils_e2e import custom_timer_with_return


@custom_timer_with_return
def train_per_epoch_supervised_selective_composite_e2e(
    args,
    device,
    outputs,
    epoch,
    logger,
    **kwargs_training,
):
    early_stopping = []
    if kwargs_training["early_stop"].early_stop:
        logger.info(f"Training of the bucket completed.")
        early_stopping.append(kwargs_training["early_stop"].early_stop)
    else:
        early_stopping.append(kwargs_training["early_stop"].early_stop)

    sub_labels = outputs

    train_sub_loader = DataLoader(
        kwargs_training["dataset"],
        batch_size=len(sub_labels),
        shuffle=False,
        collate_fn=kwargs_training["collateGdrSpkr"],
        drop_last=True,
    )

    mel_db_batch = next(iter(train_sub_loader))

    x, _, spk = mel_db_batch
    x = x.reshape(-1, args.seg_len, args.feature_dim).to(device)
    spk = spk.to(device)

    input_buffer = {"feat": x, "label": spk}

    # Train the classifier
    kwargs_training["agent"].train(
        kwargs_training["classifier"],
        kwargs_training["optimizer"],
        kwargs_training["ce_loss"],
        input_buffer,
        epoch,
        kwargs_training["filename_dir"],
    )

    out = {"early_stops_status": early_stopping}

    return out
