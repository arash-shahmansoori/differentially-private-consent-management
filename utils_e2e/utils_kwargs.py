def dataset_kwargs(
    SubDatasetGdrSpk,
    collateGdrSpkr,
    dataset,
    dataset_val,
    dataset_other=None,
    dataset_other_val=None,
):
    """Create the dictionary of dataset(s) as input arguments for the training/validation function."""
    dataset_kwargs_dict = {
        "SubDatasetGdrSpk": SubDatasetGdrSpk,
        "collateGdrSpkr": collateGdrSpkr,
        "dataset": dataset,
        "dataset_test": dataset_val,
        "dataset_val": dataset_val,
        "dataset_other": dataset_other,
        "dataset_other_val": dataset_other_val,
    }

    return dataset_kwargs_dict


def model_kwargs(agent, dvectors, classifier):
    """Create the dictionary of models as input arguments for the training function."""
    model_kwargs_dict = {"agent": agent, "dvectors": dvectors, "classifier": classifier}

    return model_kwargs_dict


def opt_kwargs(
    opt_dvec_type,
    opt_dvecs,
    opt_cls_type,
    optimizer,
    early_stop,
):
    """Create the dictionary of optimizers as input arguments for the training function."""
    opt_kwargs_dict = {
        "opt_dvec_type": opt_dvec_type,
        "opt_dvecs": opt_dvecs,
        "opt_cls_type": opt_cls_type,
        "optimizer": optimizer,
        "early_stop": early_stop,
    }

    return opt_kwargs_dict


def loss_kwargs(contrastive_loss, ce_loss):
    """Create the dictionary of losses as input arguments for the training function."""
    loss_kwargs_dict = {"contrastive_loss": contrastive_loss, "ce_loss": ce_loss}

    return loss_kwargs_dict
