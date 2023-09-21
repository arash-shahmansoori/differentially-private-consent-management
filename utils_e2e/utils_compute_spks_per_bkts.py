def compute_spks_per_bkts_storage(spks_in_buckets):

    spk_per_bkt_storage = []
    for _, s in enumerate(spks_in_buckets):
        spk_per_bkt_storage.append(len(s))

    return spk_per_bkt_storage