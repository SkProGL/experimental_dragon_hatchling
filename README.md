## Tuning & visualisation of Dragon Hatchling architecture 
https://github.com/pathwaycom/bdh

## Factual model
1. Make model stable
2. Scale moderately
3. Train longer
4. Compare perplexity
5. Only then experiment with LoRA
#### 1.0 invariant model components
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

Following tables will be used for accurate model training

#### 1.1 Single GPU testing


Dropout is always 0.1
#### Tunable hyperparameters

| Run | Layers | Emb | Heads | MLP Mult | LR   | Batch | Weight Decay | Iterations |
| --- | ------ | --- | ----- | -------- | ---- | ----- | ------------ | ----- |
| A1   | 6      | 256 | 4     | 64       | 1e-3 | 8    | 0.1          | 6k    |
| A2   | 8      | 384 | 6     | 64       | 5e-4 | 4    | 0.1          | 12k   |
| A3   | 12     | 512 | 8     | 64       | 3e-4 | 2    | 0.1          | 20k   |


#### Evaluation metrics (model output)
| Run | Train Loss | Val Loss | Perplexity | Sparsity | Latent/Layer | Time (hrs) |
| --- | ---------- | -------- | ---------- | -------- | ------------ | ---------- |
| A1 | 1.22 | 1.17 | 3.22 | 0.835 | 16384 | 31m |
| A2 | 1.17 | 1.06 | 2.88 | 0.853 | 24576 | 1h 10m |
| A3 | 1.23 | 1.11 | 3.05 | 0.865 | 32768 | 3h 41m |

Pick best run based on validation loss, but mention compute cost and diminishing returns.

#### 1.2 Follow-up sweep (informed by initial results)
#### Tunable hyperparameters

| Run | Layers | Emb | Heads | MLP Mult | LR | Batch | Weight Decay | Iterations | 
| --- | ------ | --- | ----- | -------- | -- | ----- | ------------ | ---------- |
| A4 | 8 | 384 | 6 | 128 | 5e-4 | 2 | 0.05 | 12k |
| A5 | 8 | 384 | 8 | 128 | 5e-4 | 2 | 0.05 | 12k |
| A6 | 8 | 512 | 8 | 128 | 4e-4 | 2 | 0.05 | 12k |
| A7 | 8 | 512 | 16 | 128 | 4e-4 | 2 | 0.05 | 12k |
| A8 | 8 | 512 | 8 | 256 | 4e-4 | 1 | 0.01 | 12k |
| A9 | 8 | 512 | 8 | 256 | 4e-4 | 1 | 0.1 | 12k |

#### Evaluation metrics (model output)
| Run | Train Loss | Val Loss | Perplexity | Sparsity | Latent/Layer | Time (hrs) |
| --- | ---------- | -------- | ---------- | -------- | ------------ | ---------- |
| A4 | 1.30 | 1.19 | 3.27 | 0.872 | 49152 | 57m |
| A5 | 1.30 | 1.20 | 3.33 | 0.869 | 49152 | 57m |
| A6 | 1.25 | 1.18 | 3.25 | 0.864 | 65536 | 1h 24m |
| A7 | 1.29 | 1.21 | 3.34 | 0.879 | 65536 | 1h 25m |
| A8 | 1.43 | 1.20 | 3.31 | 0.887 | 131072 | 1h 29m |
| A9 | 1.44 | 1.21 | 3.36 | 0.895 | 131072 | 1h 27m |


Pick best run based on validation loss, but mention compute cost and diminishing returns.

#### Optimization Sweep (on best model)
| Run | LR   | Batch | Weight Decay | Iterations |
| --- | ---- | ----- | ------------ | ----- |
| A7  | 3e-4 | 16    | 0.01         | Y     |
| A8  | 3e-4 | 16    | 0.05         | Y     |
| A9  | 5e-4 | 16    | 0.05         | Y     |
| A10 | 3e-4 | 16    | 0.05         | Y     |


Compare & monitor logits, with Transformer model


Where it would be applicable and answer research question
What could be optimized?
formalize research questions.
Was it quantized?
Double check hyperparameters from BDH paper.


Compare A1 and A6 on inference
- check probabilities outside logits
Reference research questions
