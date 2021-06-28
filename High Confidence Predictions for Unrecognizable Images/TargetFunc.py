import torch


def target_func(x, nn):
    # transform shape of x to (1, 1, 28, 28)
    # the second 1 represent the channel, 1 is the position u want to add to.
    out = nn(x.unsqueeze(1))
    max_prob, pred = torch.max(out, 1)
    return max_prob.detach().numpy(), pred.detach().numpy()