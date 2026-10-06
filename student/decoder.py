import student.training as t
import torch as torch


def generate_tokens(model, prompt_ids, max_tokens, temperature, p, eos_token_id, context_length, device):
    sequence = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, prompt_len)
    for _ in range(max_tokens):
        input_seq = sequence[:, -context_length:]
        logits = model(input_seq)[0,-1,:]
        temp_scaled_logits = logits/temperature
        probs = t.softmax(temp_scaled_logits, dim=-1)
        filtered_probs = top_p(probs,p)
        next_token = torch.multinomial(filtered_probs, num_samples=1)
        sequence = torch.cat([sequence, next_token.unsqueeze(0)], dim=1)
        if next_token.item() == eos_token_id:
            break
            
    return sequence[0].tolist()


def top_p(probs, p):
    values, ind = torch.sort(probs, descending=True)
    cumul_probs = torch.cumsum(values, dim=-1)
    ind_to_rem = cumul_probs - values > p
    ind_to_rem[0] = False
    values[ind_to_rem] = 0.0
    filtered_probs = torch.zeros_like(probs)
    filtered_probs[ind] = values
    filtered_probs = filtered_probs/filtered_probs.sum()
    return filtered_probs