# Implementation of language model decoding algorithms

## Nucleus Sampling with Flattened Head

This sampling algorithm is described in [Lingxi: A Diversity-aware Chinese Modern Poetry Generation System](https://aclanthology.org/2023.acl-demo.6.pdf). The core idea is illustrated by the following picture:

![Figure 3](ns_fh_figure3.png)

The algorithm is implemented in [ns_flattened_head_sampling.py](ns_flattened_head_sampling.py) as an add-on to the [top-p logit processor in transformers](https://huggingface.co/docs/transformers.js/en/api/generation/logits_process#new-topplogitswarpertopp-options).

The constructor accepts both `top_p` and `top_q` arguments:

```python
import transformers

logits_processor = transformers.generation.logits_process.LogitsProcessorList()
logits_processor.append(NucleusSamplingWithFlattenedHead(top_q=0.30, top_p=0.50))
```
Pass `logits_processor` to the model's `generation` method:

```python

tokenizer = transformers.AutoTokenizer.from_pretrained(...)
model = transformers.AutoModelForCausalLM.from_pretrained(...)

with torch.no_grad():
  out_ids = model.generate(...,
                           logits_processor=logits_processor,
                           do_sample=generation_args['do_sample'],
                           temperature=generation_args['temperature'],
                          )
```




