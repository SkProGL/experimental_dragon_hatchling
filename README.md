## Tuning & visualisation of Dragon Hatchling architecture 
https://github.com/pathwaycom/bdh

1.1 if there isn't enough data and it only occurs a few times, model doesn't fully learn and is more likely to hallucinate
1.2 fluent continuations
1.3 temperature can add more vocabulary/creativity to answers

### TODO next
Compare & monitor logits, with Transformer model

- Reference research questions
- Where it would be applicable and answer research question
- What could be optimized?
- Was it quantized?
- Double check hyperparameters from BDH paper.
- Compare A1 and A6 on inference
- check probabilities outside logits

## Factual model
1. Make model stable
2. Scale moderately
3. Train longer
4. Compare perplexity
5. Only then experiment with LoRA

# English-WIKI dataset
### 1.0 invariant model components
Training only
| Loss Function | Dropout |
|---------------|---------|
| cross-entropy | 0.1     |

Inference only
| Generation (Top-k) | Temperature |
|--------------------|-------------|
| 3           | 1.0         |

Training & inference
| Tokenization | Vocabulary | Attention | Positional Encoding | Activation | Sequence Length |
|--------------|------------|-----------|---------------------|------------|-----------------|
| Bits-per-byte   | 256        | causal    | RoPE                | ReLU       | 512             |

NOTE: Batch size is dynamically adjusted per run based on available GPU memory and effective utilization, rather than being fixed across experiments.

### 1.1 Initial sweep

##### Tunable hyperparameters

Pick best run based on validation loss, but mention compute cost and diminishing returns.

### 1.2 Follow-up sweep (informed by initial results)
##### Tunable hyperparameters

| Run | Layers | Emb | Heads | MLP Mult | LR | Batch | Weight Decay | Iterations | 
| --- | ------ | --- | ----- | -------- | -- | ----- | ------------ | ---------- |
| A1   | 6      | 256 | 4     | 64       | 1e-3 | 8    | 0.1          | 6k    |
| A2   | 8      | 384 | 6     | 64       | 5e-4 | 4    | 0.1          | 12k   |
| A3   | 12     | 512 | 8     | 64       | 3e-4 | 2    | 0.1          | 20k   |
| A4 | 8 | 384 | 6 | 128 | 5e-4 | 2 | 0.05 | 12k |
| A5 | 8 | 384 | 8 | 128 | 5e-4 | 2 | 0.05 | 12k |
| A6 | 8 | 512 | 8 | 128 | 4e-4 | 2 | 0.05 | 12k |
| A7 | 8 | 512 | 16 | 128 | 4e-4 | 2 | 0.05 | 12k |
| A8 | 8 | 512 | 8 | 256 | 4e-4 | 1 | 0.01 | 12k |
| A9 | 8 | 512 | 8 | 256 | 4e-4 | 1 | 0.1 | 12k |
| A10 | 8 | 384 | 6 | 64 | 5e-04 | 4 | 0.1 | 30k |

##### Evaluation metrics (model output)
| Run | Train Loss | Val Loss | Perplexity | Sparsity | Latent/Layer | Time (hrs) |
| --- | ---------- | -------- | ---------- | -------- | ------------ | ---------- |
| A1 | 1.22 | 1.17 | 3.22 | 0.835 | 16384 | 31m |
| A2 | 1.17 | 1.06 | 2.88 | 0.853 | 24576 | 1h 10m |
| A3 | 1.23 | 1.11 | 3.05 | 0.865 | 32768 | 3h 41m |
| A4 | 1.30 | 1.19 | 3.27 | 0.872 | 49152 | 57m |
| A5 | 1.30 | 1.20 | 3.33 | 0.869 | 49152 | 57m |
| A6 | 1.25 | 1.18 | 3.25 | 0.864 | 65536 | 1h 24m |
| A7 | 1.29 | 1.21 | 3.34 | 0.879 | 65536 | 1h 25m |
| A8 | 1.43 | 1.20 | 3.31 | 0.887 | 131072 | 1h 29m |
| A9 | 1.44 | 1.21 | 3.36 | 0.895 | 131072 | 1h 27m |
| A10 | 1.03 | 0.95 | 2.57 | 0.877 | 24576 | 2h 16m |

Pick best run based on validation loss, but mention compute cost and diminishing returns.

A1-A3 initial hyperparameters

Optimization Sweep (on best model)
A10 is based on best run A2 hyperparameters, only with increased number of iterations



### 1.3 Transformer run on same parameters
##### Tunable hyperparameters
| Run | Layers | Emb | Heads | MLP Mult | LR | Batch | Weight Decay | Iterations |
| --- | ------ | --- | ----- | -------- | -- | ----- | ------------ | ---------- |
| B1 | 6 | 256 | 4 | 64 | 1e-03 | 8 | 0.1 | 6k |
| B2 | 8 | 384 | 6 | 64 | 5e-04 | 4 | 0.1 | 12k |
| B3 | 12 | 512 | 8 | 64 | 3e-04 | 2 | 0.1 | 20k |
| B4 | 8 | 384 | 6 | 128 | 5e-04 | 2 | 0.05 | 12k |
| B5 | 8 | 384 | 8 | 128 | 5e-04 | 2 | 0.05 | 12k |
| B6 | 8 | 512 | 8 | 128 | 4e-04 | 2 | 0.05 | 12k |
| B7 | 8 | 512 | 10 (wrong, should be 16) | 128 | 4e-04 | 2 | 0.05 | 12k |
| B8 | 8 | 512 | 8 | 256 | 4e-04 | 1 | 0.01 | 12k |
| B9 | 8 | 512 | 8 | 256 | 4e-04 | 1 | 0.1 | 12k |

##### Evaluation metrics (model output)
| Run | Train Loss | Val Loss | Perplexity | Sparsity | Latent/Layer | Time (hrs) |
| --- | ---------- | -------- | ---------- | -------- | ------------ | ---------- |
| B1 | 1.23 | 1.14 | 3.12 | - | - | 2m |
| B2 | 1.21 | 1.10 | 3.00 | - | - | 3m |
| B3 | 1.35 | 1.18 | 3.26 | - | - | 5m |
| B4 | 1.32 | 1.15 | 3.16 | - | - | 3m |
| B5 | 1.32 | 1.15 | 3.16 | - | - | 3m |
| B6 | 1.32 | 1.21 | 3.34 | - | - | 3m |
| B7 | 1.32 | 1.21 | 3.34 | - | - | 3m |
| B8 | 1.46 | 1.31 | 3.70 | - | - | 3m |
| B9 | 1.51 | 1.39 | 4.03 | - | - | 3m |


# TinyStories dataset
# BDH
| Run | Layers | Emb | Heads | MLP Mult | LR | Batch | Weight Decay | Iterations |
| --- | ------ | --- | ----- | -------- | -- | ----- | ------------ | ---------- |
| C1 | 6 | 256 | 4 | 64 | 1e-03 | 8 | 0.1 | 6k |
| C2 | 8 | 384 | 6 | 64 | 5e-04 | 4 | 0.1 | 12k |
| C3 | 12 | 512 | 8 | 64 | 3e-04 | 2 | 0.1 | 20k |
| C4 | 8 | 384 | 6 | 128 | 5e-04 | 2 | 0.05 | 12k |
| C5 | 8 | 384 | 8 | 128 | 5e-04 | 2 | 0.05 | 12k |
| C6 | 8 | 512 | 8 | 128 | 4e-04 | 2 | 0.05 | 12k |
| C7 | 8 | 512 | 16 | 128 | 4e-04 | 2 | 0.05 | 12k |
| C8 | 8 | 512 | 8 | 256 | 4e-04 | 1 | 0.01 | 12k |
| C9 | 8 | 512 | 8 | 256 | 4e-04 | 1 | 0.1 | 12k |

| Run | Train Loss | Val Loss | Perplexity | Sparsity | Latent/Layer | Time (hrs) |
| --- | ---------- | -------- | ---------- | -------- | ------------ | ---------- |
| C1 | 0.73 | 0.69 | 2.00 | 0.827 | 16384 | 24m |
| C2 | 0.67 | 0.61 | 1.85 | 0.837 | 24576 | 55m |
| C3 | 0.68 | 0.65 | 1.91 | 0.865 | 32768 | 1h 43m |
| C4 | 0.75 | 0.72 | 2.05 | 0.865 | 49152 | 57m |
| C5 | 0.75 | 0.71 | 2.04 | 0.862 | 49152 | 56m |
| C6 | 0.74 | 0.68 | 1.97 | 0.856 | 65536 | 1h 25m |
| C7 | 0.76 | 0.68 | 1.98 | 0.870 | 65536 | 1h 24m |
| C8 | 0.84 | 0.77 | 2.16 | 0.863 | 131072 | 1h 28m |
| C9 | 0.85 | 0.78 | 2.19 | 0.882 | 131072 | 1h 28m |


