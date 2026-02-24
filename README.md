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
| A1   | 6      | 256 | 4     | 64       | 1e-3 | 16    | 0.1          | 6k    |
| A2   | 8      | 384 | 6     | 64       | 5e-4 | 16    | 0.1          | 12k   |
| A3   | 12     | 512 | 8     | 64       | 3e-4 | 16    | 0.1          | 20k   |


#### Evaluation metrics (model output)
| Run | Train Loss | Val Loss | Perplexity | Time (hrs) |
| --- | ---------- | -------- | ---------- | ---------- |
| A1  | a1 | b1 | c1 | d1 |
| A2  | a2 | b2 | c2 | d2 |
| A3  | a3 | b3 | c3 | d3 |

Pick best run based on validation loss, but mention compute cost and diminishing returns.

#### 1.2 Multi-GPU testing
#### Tunable hyperparameters

| Run | Layers | Emb | Heads | MLP Mult | LR   | Batch | Weight Decay | Iterations |
| --- | ------ | --- | ----- | -------- | ---- | ----- | ------------ | ----- |
| B1   | 6      | 256 | 4     | 32       | 1e-3 | 16    | 0.05         | 15k   |
| B2   | 6      | 256 | 4     | 64       | 1e-3 | 16    | 0.05         | 15k   |
| B3   | 8      | 384 | 6     | 48       | 5e-4 | 16    | 0.05         | 20k   |
| B4   | 8      | 384 | 6     | 64       | 5e-4 | 16    | 0.05         | 20k   |
| B5   | 12     | 512 | 8     | 48       | 3e-4 | 16    | 0.05         | 30k   |
| B6   | 12     | 512 | 8     | 64       | 3e-4 | 16    | 0.05         | 30k   |


#### Evaluation metrics (model output)
| Run | Train Loss | Val Loss | Perplexity | Time (hrs) |
| --- | ---------- | -------- | ---------- | ---------- |
| B1  | a1 | b1 | c1 | d1 |
| B2  | a2 | b2 | c2 | d2 |
| B3  | a3 | b3 | c3 | d3 |
| B4  | a4 | b4 | c4 | d4 |
| B5  | a5 | b5 | c5 | d5 |
| B6  | a6 | b6 | c6 | d6 |

Pick best run based on validation loss, but mention compute cost and diminishing returns.

#### Optimization Sweep (on best model)
| Run | LR   | Batch | Weight Decay | Iterations |
| --- | ---- | ----- | ------------ | ----- |
| B7  | 3e-4 | 16    | 0.01         | Y     |
| B8  | 3e-4 | 16    | 0.05         | Y     |
| B9  | 5e-4 | 16    | 0.05         | Y     |
| B10 | 3e-4 | 16    | 0.05         | Y     |
