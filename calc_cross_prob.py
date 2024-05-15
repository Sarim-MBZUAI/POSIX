import torch 

def response_prob(prompt, response, tokenizer, model):
    input_text = prompt + ' ' + response

    #get tokens
    prompt_tokens = tokenizer.encode(prompt)
    n_prompt_tokens = len(prompt_tokens)

    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    
    response_tokens = input_ids[0, n_prompt_tokens:]
    
    total_tokens = input_ids.shape[-1]
    n_response_tokens = total_tokens - n_prompt_tokens
    
    #get logits
    logits = model(input_ids).logits
  
    response_logits = logits[:,n_prompt_tokens-1:total_tokens-1, :]
    response_probs = torch.softmax(response_logits, dim=-1)

    final_prob = 1.0
    for i in range(n_response_tokens):
        final_prob = final_prob*(response_probs[0, i, response_tokens[i]].item())

    #product of small probabilities may lead to underflow
    #should we do summation of log probs instead?
    # log_response_probs = torch.log(response_probs)
    # final_prob = 0.0
    # for i in range(n_response_tokens):
    #     final_prob = final_prob + log_response_probs[0, i, response_tokens[i]]
    
    return final_prob


def calc_prob_dist(n1, n2, tokenizer, model):
    """
    n1 and n2 are two tuples
    First element of the tuples is the input prompt
    Second element of the tuple is the output
    """
    dist = 1 - 0.5*(response_prob(n1[0], n1[1], tokenizer, model) + response_prob(n2[0], n2[1], tokenizer, model))
    return dist
