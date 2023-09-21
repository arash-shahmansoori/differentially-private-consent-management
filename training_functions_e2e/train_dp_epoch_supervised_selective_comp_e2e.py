from utils_e2e import custom_timer_with_return


@custom_timer_with_return
def train_dp_per_epoch_supervised_selective_composite_e2e(
    input_data,
    logger,
    epoch,
    **kwargs_training,
):
    early_stopping = []
    if kwargs_training["early_stop"].early_stop:
        logger.info(f"Training of the bucket completed.")
        early_stopping.append(kwargs_training["early_stop"].early_stop)
    else:
        early_stopping.append(kwargs_training["early_stop"].early_stop)

    input_buffer = {"feat": input_data["x"], "label": input_data["y"]}

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
