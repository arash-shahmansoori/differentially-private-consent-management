import torch


class EarlyStoppingCustomComposite:
    """Early stops the training if accuracy is in a certain range."""

    def __init__(self, args, mode="non-dp"):
        """
        args:
                necessary arguments for early stopping including =>
                patience (int): How long to wait after last time improved; Default: 5
                verbose (bool): If True, prints a message for each improvement; Default: False
                min_delta (float): Minimum change in the monitored quantity to qualify as an improvement; Default: 0

        """
        self.patience = args.patience_stopping
        self.verbose = args.verbose_stopping
        self.delta = args.min_delta

        if mode == "dp":
            self.threshold_val_acc = args.threshold_val_acc_dp
        else:
            self.threshold_val_acc = args.threshold_val_acc

        # Initializations
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, acc, epoch):
        score = acc

        if self.best_score is None:
            self.best_score = score
        elif acc - torch.tensor(self.threshold_val_acc) >= self.delta:
            self.counter += 1

            if self.counter < self.patience and self.early_stop == False:
                print(
                    f"Early stopping at ep:{epoch}, cnt:{self.counter}/{self.patience}"
                )
            elif self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            self.counter = 0
