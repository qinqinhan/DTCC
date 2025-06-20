import torch
def target_distribution(q):
    weight = (q ** 2.0) / torch.sum(q, 0)
    return (weight.t() / torch.sum(weight, 1)).t()


def forward_prob(q_i):
    q_i = target_distribution(q_i)

    p_i = q_i.sum(0).view(-1)
    p_i /= p_i.sum()
    ne_i = (p_i * torch.log(p_i)).sum()



    entropy = ne_i

    return entropy
