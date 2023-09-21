import torch
import logging


# Functions for selection of buckets with/without overlaps
def cor_seq_counter_list(N, s, stride):
    x = []
    y = []

    for i in range(0, N - 1, stride):
        for j in range(0, N):
            if j >= i and j < i + s:
                x.append(j)
        if len(x) == s:
            y.append(x)
            x = []
    return y


def cor_seq_counter(N, s, stride):
    x = []
    y = []

    for i in range(0, N - 1, stride):
        for j in range(0, N):
            if j >= i and j < i + s:
                x.append(j)
        if len(x) == s:
            y.append(torch.tensor(x))
            x = []
    return torch.stack(y).shape[0], torch.stack(y)
