Part 1: GPTQ on OPT-125m (WikiText-2, 4-bit, group size 128)

Metric	Base (fp16)	GPTQ	Change
Perplexity	43.99	46.63	+6.0%
p50 latency	38.85 ms	44.73 ms	0.87× speed (slower)
p95 latency	64.85 ms	72.84 ms	slower
Peak GPU memory	0.277 GB	0.152 GB	45% lower
Size on disk	250.5 MB	125.1 MB	2.0× smaller

Part 2 and 3: BERT-base teacher vs best DistilBERT student (AG News, T = 10, α = 0.5)

Metric	Teacher	Student	Change
Test accuracy	94.68%	94.71%	+0.03 points
p50 latency	9.64 ms	5.04 ms	1.91× faster
p95 latency	10.19 ms	8.13 ms	faster
Peak GPU memory	0.880 GB	0.272 GB	69% lower
Size on disk	438.0 MB	267.8 MB	1.64× smaller

Sweeps (test accuracy)

Temperature (α = 0.5)	T = 1	T = 2	T = 5	T = 10
Accuracy	94.50%	94.45%	94.64%	94.71%
Alpha (T = 10)	0.25	0.5	0.75
Accuracy	94.64%	94.71%	94.61%

Per-class accuracy

Class	Teacher	Student	Difference
World	95.63%	95.42%	−0.21
Sports	98.79%	98.89%	+0.11
Business	91.32%	91.11%	−0.21
Sci/Tech	93.00%	93.42%	+0.42
