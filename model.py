import time
from typing import Dict, List, Any, Union
import numpy as np
import torch
from transformers import AutoTokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer


# --- Speculative Sampling Function ---
def speculative_sample(primal_probs, draft_probs, draft_token):
    """Speculative sampling to accept/reject draft guess and adjust distribution."""
    if isinstance(draft_token, torch.Tensor):
        draft_token = draft_token.item()

    if primal_probs.dim() == 2 and primal_probs.size(0) == 1:
        p_x = primal_probs[0, draft_token].item()
        q_x = draft_probs[0, draft_token].item()
    else:
        raise ValueError(f"Unexpected shape for primal probs: {primal_probs.shape}")

    r = np.random.uniform(0, 1)
    if r <= p_x / q_x:
        return draft_token, None
    else:
        adjusted_probs = torch.max(primal_probs[0] - draft_probs[0], torch.zeros_like(draft_probs[0]))
        adjusted_probs = adjusted_probs / adjusted_probs.sum()
        new_token = torch.multinomial(adjusted_probs, num_samples=1).item()
        return None, new_token

class HFModelEngine:
    def __init__(self, model_name: str, batch_size: int, device):
        self.device = device
        self.model = self.load_model(model_name)
        self.tokenizer = self.load_tokenizer()
        # self._queue: asyncio.Queue = asyncio.Queue()

    def load_model(self, model_name):
        """Load the specified model and move it to the given device."""
        model = GPT2LMHeadModel.from_pretrained(model_name)
        model.eval()
        model.to(self.device)
        return model

    def load_tokenizer(self):
        """Load shared tokenizer."""
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def make_inputs(self, prompt):
        if isinstance(prompt, str):
            return self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        elif isinstance(prompt, List):
            return torch.tensor(prompt, device=self.device)
        else:
            return prompt
         
    def draft_generate(self, prompt, max_tokens, proposal_length, temperature):
        input_ids = self.make_inputs(prompt)
        prompt_length = input_ids.shape[1]
        # Compute gamma based on max_tokens: max_tokens - prompt_length - 1 (for the extra token)
        gamma = max(0, max_tokens - prompt_length - 1)  # Subtract 1 for the extra token
        gamma = min(gamma, proposal_length) if proposal_length is not None else gamma # check if there is a constraint in proposal_length
        draft_guesses = []
        current_ids = input_ids.clone()
        for _ in range(gamma):
            draft_probs, draft_token = self.get_distribution(current_ids, temperature)
            draft_guesses.append((draft_token, draft_probs.tolist()))
            current_ids = torch.cat([current_ids, torch.tensor([[draft_token]], device=self.device)], dim=1)
        
        # Simulate processing time (1.0 second per prompt)
        time.sleep(1.0)
        return {"input_ids": input_ids.tolist(), "guesses": draft_guesses}

    # --- Probability Distribution Function ---
    def get_distribution(self, input_ids, temperature=1.0):
        """Get probability distribution over next token from model."""
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[:, -1, :]
            if temperature == 0:
                probs = torch.softmax(logits, dim=-1)
                token_id = torch.argmax(probs, dim=-1).item()
                return probs, token_id
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                token_id = torch.multinomial(probs, num_samples=1).item()
                return probs, token_id


    # --- Primal Model Serving ---
    def verify_generate(self, input_ids, draft_guesses, temperature, max_tokens=30, gen_tokens=0):
        """Serve as primal model to verify or reject tokens in FIFO order from multiple queues."""
        process_time = time.ctime()
        input_ids = self.make_inputs(input_ids)

        accepted_tokens = []
        current_ids = input_ids.clone()
        for draft_token, draft_probs in draft_guesses:
            draft_probs = torch.tensor(draft_probs, device=self.device)
            primal_probs, _ = self.get_distribution(current_ids, temperature=temperature)
            accepted_token, rejected_token = speculative_sample(primal_probs, draft_probs, draft_token)
            if accepted_token is not None:
                accepted_tokens.append(accepted_token)
                current_ids = torch.cat([current_ids, torch.tensor([[accepted_token]], device=self.device)], dim=1)
            else:
                accepted_tokens.append(rejected_token)
                current_ids = torch.cat([current_ids, torch.tensor([[rejected_token]], device=self.device)], dim=1)
                break

        # If all speculative tokens are accepted, add one more token
        if len(accepted_tokens) == len(draft_guesses):
            primal_probs, extra_token = self.get_distribution(current_ids, temperature=temperature)
            accepted_tokens.append(extra_token)
            current_ids = torch.cat([current_ids, torch.tensor([[extra_token]], device=self.device)], dim=1)

        # Ensure the final output does not exceed max_tokens
        final_tokens = self.tokenizer.encode(self.tokenizer.decode(current_ids[0], skip_special_tokens=True), return_tensors="pt")
        if final_tokens.shape[1] > max_tokens:
            # Truncate to max_tokens
            truncated_tokens = final_tokens[:, :max_tokens]
            served_output = self.tokenizer.decode(truncated_tokens[0], skip_special_tokens=True)
        else:
            served_output = self.tokenizer.decode(current_ids[0], skip_special_tokens=True)
        
        gen_tokens += len(accepted_tokens)
        is_finished = max_tokens - gen_tokens <= 0
        return current_ids.tolist(), served_output, accepted_tokens, gen_tokens, is_finished



# from vllm import AsyncLLMEngine, SamplingParams, EngineArgs
# from vllm.inputs import TokensPrompt
# # vLLM engine setup (shared across workers if possible, or per worker)
# class VLLMModelEngine:
#     def __init__(self, model_name: str, batch_size: int):
#         engine_args = EngineArgs(
#             model=model_name,
#             max_num_seqs=batch_size,
#             max_model_len=2048,
#             disable_log_stats=False,
#         )
#         engine_args.disable_log_requests = False
#         self.engine = AsyncLLMEngine.from_engine_args(engine_args)
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)

#     def make_inputs(self, prompt_text):
#         prompt_token_ids = self.tokenizer.encode(prompt_text)
#         return TokensPrompt(prompt_token_ids=prompt_token_ids)
    
#     def get_sample_params(self, kwargs):
#         return SamplingParams(**kwargs)