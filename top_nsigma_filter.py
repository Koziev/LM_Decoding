import transformers
import torch


# Straightforward implementation of "Top-nσ sampling" described in
# https://aclanthology.org/2025.acl-long.528.pdf "Top-nσ: Eliminating Noise in Logit Space for Robust Token Sampling of LLM"
class TopNSigmaFilter(transformers.generation.logits_process.LogitsProcessor):
    def __init__(self, n_sigma: float, filter_value: float = -float("Inf")):
        assert(n_sigma>0)
        self.n_sigma = n_sigma
        self.filter_value = filter_value

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        row_maxs = scores.max(dim=1).values
        row_stds = scores.std(dim=1)
        thresholds = row_maxs - self.n_sigma * row_stds

        # Zero logits below the threshold
        # Reshape thresholds for broadcasting [10, 1] -> compares with each element in row
        indices_to_remove = scores < thresholds.unsqueeze(1)
        scores_processed = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores_processed
