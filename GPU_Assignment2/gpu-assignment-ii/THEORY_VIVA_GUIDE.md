# DATA 266 — lecture alignment and viva preparation

## BERT and fine-tuning
Lecture: BERT is a bidirectional Transformer encoder, pre-trained using masked-language modeling (and original BERT's next-sentence prediction). Here the pretrained encoder is **fine-tuned for supervised sentiment classification**, not retrained using MLM. Explain the classification head, cross-entropy loss, tokenization, and why the test split stays untouched until evaluation.

## Attention and memory
Self-attention has a quadratic token-pair interaction with sequence length in standard dense attention. Training memory consists of model weights, gradients, optimizer states, activations, and buffers. Report peak allocated memory separately from total GPU process usage.

## AMP / FP16 / BF16
Mixed precision lowers some activation and computation precision. FP16 may require gradient scaling; BF16 has greater exponent range. Do not assume 50% total training-memory savings because optimizer states and other tensors remain.

## Checkpointing
Discard intermediate activations during the forward pass and recompute them in backward. It trades compute time for activation memory. Quantify measured memory savings and throughput loss.

## Gradient accumulation
Use a small microbatch and combine gradients before an optimizer step. Effective batch = microbatch × accumulation steps (except a partial final step). It does not make the physical batch fit more examples at once. Compare equal optimizer steps or explain any mismatch.

## 8-bit optimizer vs weight quantization
An 8-bit optimizer compresses **optimizer state**. This is not equivalent to 8-bit model weight quantization, nor LoRA/QLoRA. Report which tensor class is reduced and accuracy/throughput trade-offs.

## vLLM and KV cache
BERT is encoder-only, so benchmark a separate supported causal language model for vLLM. During autoregressive decoding, caching past keys/values avoids recomputing prior-token attention projections; cache usage rises with active sequence length and concurrency. Process-level GPU memory is not a direct isolated KV-cache measurement. Request latency and throughput are distinct metrics; synchronous batch-completion times are only a proxy for independent request latency.

## Answer prompts (fill in after experiments)
1. Which batch gave the best throughput before OOM and why?
2. When did validation accuracy stop improving and what was the cost of later epochs?
3. Which technique saved the most memory per unit throughput lost?
4. Why does checkpointing often slow training?
5. Why is gradient accumulation not identical to increasing physical batch size?
6. Why do throughput and per-request latency move differently as concurrent requests increase?
7. Which context lengths fit batch 8 and how did you measure KV-cache memory?
8. Why is your selected configuration preferable for your GPU?

Write answers from actual measurements; never invent performance figures.
