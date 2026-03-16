## Tuning & visualisation of Dragon Hatchling architecture 
https://github.com/pathwaycom/bdh

## Factual model
1. Make model stable
2. Scale moderately
3. Train longer
4. Compare perplexity
5. Only then experiment with LoRA

NOTE: batch size is 16 due to limited RAM

Following tables will be used for accurate model training

#### 1.1 Single GPU testing
#### Tunable hyperparameters

| Run | Layers | Emb | Heads | MLP Mult | LR   | Batch | Weight Decay | Iterations |
| --- | ------ | --- | ----- | -------- | ---- | ----- | ------------ | ----- |
| A1   | 6      | 256 | 4     | 64       | 1e-3 | 8    | 0.1          | 6k    |
| A2   | 8      | 384 | 6     | 64       | 5e-4 | 4    | 0.1          | 12k   |
| A3   | 12     | 512 | 8     | 64       | 3e-4 | 2    | 0.1          | 20k   |


#### Evaluation metrics (model output)
| Run | Train Loss | Val Loss | Perplexity | Time (hrs) |
| --- | ---------- | -------- | ---------- | ---------- |
| A1  | 1.5126 | 1.4788 | 7.0657 | 31min |
| A2  | 1.4167 | 1.4043 | 5.5118 | 1h 10min |
| A3  | 1.5598 | 1.5585 | 6.3409 | 3h 41min |

Pick best run based on validation loss, but mention compute cost and diminishing returns.

#### 1.2 Multi-GPU tuning
#### Tunable hyperparameters

| Run | Layers | Emb | Heads | MLP Mult| LR   | Batch | Weight Decay | Iterations|
| --- | ------ | --- | ----- | --- | ---- | ----- | ------------ | ---- |
| A4  | 8      | 384 | 6      | 128 | 5e-4 | 8    | 0.05         | 12k  |
| A5  | 8      | 384 | 8      | 128 | 5e-4 | 8    | 0.05         | 12k  |
| A6  | 8      | 512 | 8      | 128 | 4e-4 | 8    | 0.05         | 12k  |
| A7  | 8      | 512 | 10     | 128 | 4e-4 | 8    | 0.05         | 12k  |
| A8  | 8      | 512 | 8      | 256 | 4e-4 | 8    | 0.01         | 12k  |
| A9  | 8      | 512 | 8      | 256 | 4e-4 | 8    | 0.1          | 12k  |

#### Evaluation metrics (model output)
| Run | Train Loss | Val Loss | Perplexity | Time (hrs) |
| --- | ---------- | -------- | ---------- | ---------- |
| A4  | a1 | b1 | c1 | d1 |
| A5  | a2 | b2 | c2 | d2 |
| A6  | a3 | b3 | c3 | d3 |
| A7  | a4 | b4 | c4 | d4 |
| A8  | a5 | b5 | c5 | d5 |
| A9  | a6 | b6 | c6 | d6 |

Pick best run based on validation loss, but mention compute cost and diminishing returns.

#### Optimization Sweep (on best model)
| Run | LR   | Batch | Weight Decay | Iterations |
| --- | ---- | ----- | ------------ | ----- |
| A7  | 3e-4 | 16    | 0.01         | Y     |
| A8  | 3e-4 | 16    | 0.05         | Y     |
| A9  | 5e-4 | 16    | 0.05         | Y     |
| A10 | 3e-4 | 16    | 0.05         | Y     |
