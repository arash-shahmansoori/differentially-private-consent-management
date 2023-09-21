import torch

# Functions for performance evaluation
def num_correct(outputs, labels, topk=1):
    probs, preds = outputs.topk(k=topk, dim=1)
    preds = preds.view(-1)
    correct = preds.eq(labels.expand_as(preds))

    p_pr = []
    n_pr = []
    for i, cor in enumerate(correct.tolist()):
        if cor == True:
            p_pr.append(outputs[i, preds[i]].tolist())
            n_pr.append(outputs[i, ~preds[i]].tolist())

    correct = correct.sum()
    return correct, torch.tensor(p_pr), torch.tensor(n_pr)