from dataclasses import dataclass, field


@dataclass
class HyperParamsDP:
    """A data-class that maintains the hyper-parameters to create
    differentially private features for the composite buckets/components.

    Definition:
        composite bucket/component: a composite bucket/component containing sub-buckets.
        Sub-buckets may contain:
        - sub-buckets if they are composite buckets/components
        - leaves
    """

    # The component containing leaf/leaves
    buckets: list = field(default_factory=lambda: [0, 1])  # bucket with public data
    buckets_dp: list = field(
        default_factory=lambda: [0, 1]
    )  # bucket including private data

    # Parameter for box-constraint of the component
    c: float = 1e-4

    # Confidence metric for the component's leaves
    kappa: float = 0.0

    # Number of steps to update adv features for the component
    num_steps: int = 100

    # Learning rates
    lr_cls: float = 1e-2
    lr_cls_dp: float = 1e-2
    lr_memb_cls: float = 1e-2

    # Parameters for differential privacy
    eps: float = 0.2
    delta: float = 1e-5
    clipping_norm: float = 1
    sigma: int = 0.8

    # Determine the states of the model chechpoints
    model_str: str = "model"
    opt_str: str = "optimizer"
    contloss_str: str = "criterion"
    start_epoch: str = "epoch"
