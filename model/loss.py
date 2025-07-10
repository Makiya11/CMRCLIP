import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

class NormSoftmaxLoss(nn.Module):
    def __init__(self, temperature=0.05):
        super().__init__()
        self.temperature = temperature

    def forward(self, text_projection, video_projection):
        x = sim_matrix(text_projection, video_projection)
        "Assumes input x is similarity matrix of N x M \in [-1, 1], computed using the cosine similarity between normalised vectors"
        i_logsm = F.log_softmax(x/self.temperature, dim=1)
        j_logsm = F.log_softmax(x.t()/self.temperature, dim=1)

        # sum over positives
        idiag = torch.diag(i_logsm)
        loss_i = idiag.sum() / len(idiag)

        jdiag = torch.diag(j_logsm)
        loss_j = jdiag.sum() / len(jdiag)
        return - loss_i - loss_j
    
class ClipLoss(nn.Module):
    def __init__(self, logit_scale_init=np.log(1/0.07)):
        super().__init__()
        # logit_scale is a *learnable* parameter
        self.logit_scale = nn.Parameter(torch.tensor(logit_scale_init, dtype=torch.float32))

    def forward(self, txt, vid):
        # ‑‑ both txt & vid must be ℓ2‑normalised
        logits = txt @ vid.T                      # (B, B) cosine similarities
        scale  = self.logit_scale.exp().clamp(max=100)   # optional clamp
        logits = logits * scale

        labels = torch.arange(logits.size(0), device=logits.device)

        # symmetric cross‑entropy
        loss_i = F.cross_entropy(logits, labels)
        loss_j = F.cross_entropy(logits.T, labels)
        return (loss_i + loss_j) / 2
    
def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt
