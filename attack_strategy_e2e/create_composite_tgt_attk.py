def create_tgt_composite(TgtLeaf, tgt_composite, buckets):
    """Create composite targets for different buckets.
    Args:
        - TgtLeaf (class reference): target labels for the leaf component.
        - tgt_composite (object): target labels for the composite component.
        - buckets: list of buckets.

    Return:
        - tgt_composite: composite targets for different buckets.
    """

    for _, _ in enumerate(buckets):

        tgt_leaf = TgtLeaf()
        tgt_composite.add(tgt_leaf)

    return tgt_composite


def create_attk_composite(AttkLeaf, attk_composite, buckets):
    """Create the composite attack on different buckets.
    Args:
        - AttkLeaf (class reference): attack on the leaf component.
        - attk_composite (object): attack on the composite component.
        - buckets: list of buckets.

    Return:
        - attk_composite: composite attack on different buckets.
    """

    for _, _ in enumerate(buckets):

        attk_leaf = AttkLeaf()
        attk_composite.add(attk_leaf)

    return attk_composite
