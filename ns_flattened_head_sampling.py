import transformers
import torch


# Straightforward implementation of NS-FlattenHead sampling algirithm, introduced in https://aclanthology.org/2023.acl-demo.6.pdf
class NucleusSamplingWithFlattenedHead(transformers.generation.logits_process.LogitsProcessor):
    def __init__(self, top_q: float, top_p: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        assert(0 < top_q < 1)
        self.top_q = top_q

        top_p = float(top_p)
        if top_p < 0 or top_p > 1.0:
            raise ValueError(f"`top_p` has to be a float > 0 and < 1, but is {top_p}")
        if not isinstance(min_tokens_to_keep, int) or (min_tokens_to_keep < 1):
            raise ValueError(f"`min_tokens_to_keep` has to be a positive integer, but is {min_tokens_to_keep}")
        self.top_p = top_p

        assert( top_q <= top_p )

        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        sorted_logits, sorted_indices = torch.sort(scores, descending=False)

        probs = sorted_logits.softmax(dim=-1)
        cumulative_probs = probs.cumsum(dim=-1)

        # Now flatten the top-q subset logits for each row

        items_in_topq = cumulative_probs > (1 - self.top_q)  # true для элементов scores после сортировки, которые входят в top-q (по строкам)

        # Sum up the values in top-q subsets
        topq_mass_per_row = torch.where(items_in_topq, sorted_logits, 0).sum(dim=1)

        # Number of tokens in top-q subsets per row
        num_topq_tokens = items_in_topq.sum(dim=1)

        flattened_values = topq_mass_per_row / num_topq_tokens

        items_to_flatten = items_in_topq.scatter(dim=1, index=sorted_indices, src=items_in_topq)  # true для элементов scores, которые входят в top-q (по строкам)
        scores = torch.where(items_to_flatten, flattened_values.unsqueeze(1), scores)

        # --------------------------------------------------
        # Apply top-p, assigning the tails to -inf
        # --------------------------------------------------

        # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs <= (1 - self.top_p)
        # Keep at least min_tokens_to_keep
        sorted_indices_to_remove[..., -self.min_tokens_to_keep :] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        scores_processed = scores.masked_fill(indices_to_remove, self.filter_value)

        return scores_processed
