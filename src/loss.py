import torch
from torch.nn import functional as F

def transE_loss(true_score, head_false_score, tail_false_score, gamma, p):
    # head_changed = F.relu(gamma + true_score.norm(p=p) - head_false_score.norm(p=p))
    # tail_changed = F.relu(gamma + true_score.norm(p=p) - tail_false_score.norm(p=p))
    
    head_changed = gamma + true_score.norm(p=p) - head_false_score.norm(p=p)
    tail_changed = gamma + true_score.norm(p=p) - tail_false_score.norm(p=p)
    return head_changed + tail_changed