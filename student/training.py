import torch
import torch.optim as optim
from einops import einsum
import math

def softmax(x, dim):
    max, _ = torch.max(x,dim=dim, keepdim=True)
    # print("Max value: ", max)
    # print("X now:", x)
    x = x - max
    # print("X later:", x)
    exp = torch.exp(x)
    # print("Exp:", exp)
    sum_exp = torch.sum(exp, dim=dim,keepdim=True)
    # print("Sum:", sum_exp)
    return exp / sum_exp


def scaled_dot_product_attention(Q, K, V, mask=None):
    scores = einsum(Q, K, "... n d_k, ... m d_k -> ... n m")
    d_k = Q.shape[-1]
    scores = scores * (1 / math.sqrt(d_k))
    # print("Scaled score:", scores)

    if mask is not None:
        masked_scores = torch.where(mask, scores, float("-inf"))

    attention_probs = softmax(masked_scores, -1)
    # print("Attention probs:", attention_probs)
    attention = einsum(attention_probs, V, "... n m, ... m d_v -> ... n d_v")
    # print("Final attention tensor: ", attention)
    return attention


def cross_entropy(logits, target):
    logits_flat = logits.reshape(-1, logits.shape[-1])
    target_flat = target.reshape(-1)

    max_vals, _ = torch.max(logits_flat,dim=-1,keepdim=True)
    log_sum_exp = max_vals + torch.log(torch.sum(torch.exp(logits_flat - max_vals),dim=-1,keepdim=True))
    correct_logits = torch.gather(logits_flat,dim=-1,index=target_flat.unsqueeze(-1))
    loss = log_sum_exp - correct_logits
    return loss.mean()


def learning_rate_schedule(t, lr_min, lr_max, Tw, Tc):
    if t<Tw:
        lr_t = (t/Tw) * lr_max
    elif t <= Tc and t >= Tw:
        lr_t = lr_min + 0.5*(1 + math.cos(((t-Tw)/(Tc-Tw))*math.pi))*(lr_max-lr_min)
    elif t>Tc:
        lr_t = lr_min
    return lr_t    


def gradient_clipping(params, max_l2_norm, eps=1e-6):
    l2_norm = 0
    for param in params:
        if param.grad is None: continue
        l2_norm += (param.grad **2).sum()
    
    l2_norm = math.sqrt(l2_norm)
    if l2_norm<max_l2_norm: return
    
    for param in params:
        if param.grad is None: continue
        param.grad *= (max_l2_norm/(l2_norm+eps))


def get_batch(x, batch_size, context_length, device):
    start_indices = torch.randint(low=0, high=len(x)-context_length, size=(batch_size,))
    input = []
    target = []
    for i in start_indices:
        input.append(torch.tensor(x[i:i+context_length],dtype=torch.long))
        target.append(torch.tensor(x[i+1:i+context_length+1],dtype=torch.long))
    input_tensor = torch.stack(input,dim=0).to(device)
    target_tensor = torch.stack(target,dim=0).to(device)
    return (input_tensor,target_tensor)


class AdamW(optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9,0.999), eps=1e-8, weight_decay=0.01):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr,"betas": betas,"eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            beta1 = group['betas'][0]
            beta2 = group['betas'][1]
            eps = group['eps']
            weight_decay = group['weight_decay']

            for param in group['params']:
                if param.grad is None: continue
                state = self.state[param]
                t = state.get('t',1)
                m = state.get('m', torch.zeros_like(param.data))
                v = state.get('v', torch.zeros_like(param.data))
                grad = param.grad.data

                m = beta1*m + (1-beta1)*grad
                v = beta2*v + (1-beta2)*(torch.pow(grad,2))

                lr_t = lr * (math.sqrt(1 - math.pow(beta2,t)) / (1 - math.pow(beta1,t)))
                param.data -= lr_t * (m / (torch.sqrt(v) + eps))

                param.data -= lr * weight_decay * param.data
                state['t'] = t + 1
                state['m'] = m
                state['v'] = v
            return loss


