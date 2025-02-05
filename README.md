# Neural-Decoding-Project
---
### Abstract

# Neural Decoding Project

This project processes neural spike data and trains a neural network model to predict cursor velocity based on neural activity.

## Project Structure

- `data_loader.py` - Loads neural data from a remote source.
- `preprocessing.py` - Processes spike times and aligns them with indices.
- `model.py` - Defines the neural network architecture.
- `train.py` - Trains the neural decoder model and evaluates performance.
- `utils.py` - Contains helper functions for data normalization and conversion.

## Installation

```sh
pip install -r requirements.txt
```

## Usage

1. **Load Data**
   ```sh
   python data_loader.py
   ```
2. **Preprocess Data**
   ```sh
   python preprocessing.py
   ```
3. **Train Model**
   ```sh
   python train.py
   ```

## Requirements
See `requirements.txt` for dependencies.




# Neural Spike-Based Cursor Velocity Prediction with Transformers

## Overview
This project implements a Transformer-based model to predict the cursor velocity of a monkey based on its neural spike activity. The dataset consists of neural recordings from 71 channels sampled at ~100 Hz, with corresponding cursor velocity data. The goal is to model the relationship between neural activity and movement dynamics using a **decoder-only Transformer**, implemented from scratch.

## Dataset
- **Neural Data:**
  - `spike_times`: (319409,)
  - `spike_times_index`: (71,)
  
- **Cursor Velocity Data:**
  - `timestamps`: (66321,)
  - `data`: (66321, 2) → `[Vx, Vy]`

## Preprocessing Pipeline
1. **Binning Neural Data:**
   - Spikes are binned into 100ms windows, each containing 10 bins of 10ms.
   
2. **Removing Abnormal Velocities:**
   - Data points where \(|Vx| > 40\) or \(|Vy| > 40\) are filtered out.
   
3. **Tokenization and Temporal Embedding:**
   - Neural spikes are tokenized into numerical sequences.
   - Temporal embeddings are applied before passing data into the Transformer.

## Model Architecture
The model is a **decoder-only Transformer**, implemented from scratch. It consists of:
- **Multi-Head Self-Attention** (with learnable heads)
- **Feedforward Expansion Layer** (scales feature dimensions)
- **Positional Encoding**
- **Stacked Transformer Blocks**

### Hyperparameters
| Parameter           | Value  |
|--------------------|--------|
| Embedding Size     | 200    |
| Number of Layers  | 3      |
| Heads             | 5      |
| Dropout Rate      | 0.3    |
| Forward Expansion | 4      |
| Input Length      | 100    |
| Batch Size        | 64     |
| Epochs            | 40     |
| Learning Rate     | 1e-4   |

## Training and Evaluation
- **Loss Function:** Mean Squared Error (MSE)
- **Optimizer:** Adam
- **Metrics:**
  - Mean Squared Error (MSE)
  - R² Score

### Training Results (Example)
```
Epoch 1/40, Loss: 46.94
Epoch 10/40, Loss: 22.39
Epoch 20/40, Loss: 11.72
Epoch 30/40, Loss: 5.78
Epoch 40/40, Loss: 2.87
Test MSE: 61.37, R² Score: -0.39
```

## Challenges and Next Steps
🔴 **Current Issue:** Test MSE is high, and R² is negative, indicating poor generalization.
✅ **Possible Improvements:**
- Hyperparameter tuning (heads, embedding size, forward expansion)
- Alternative loss functions (e.g., Huber Loss)
- Data augmentation or regularization
- Attention visualization to understand feature importance

## How to Run
### Install Dependencies
```bash
pip install numpy torch scikit-learn matplotlib
```

### Train the Model
```bash
python train.py
```

### Evaluate
```bash
python evaluate.py
```

## Contributors
- **[Your Name]** – Model Implementation, Preprocessing
- **[Your Collaborator]** – Data Analysis, Report Writing

## License
MIT License (Feel free to modify and use)


