

**Data Preparation**
- Train/validation/test split via `train_test_split` with stratification (preserves class balance across splits)
- Mel Spectrogram transformation - converts raw audio to a frequency representation the model can learn from
- Amplitude to DB conversion - log-scales the spectrogram for better dynamic range

**Model**
- `GenreCNN` - a Convolutional Neural Network for classifying audio into genres

**Training**
- `AdamW` optimizer - gradient-based optimization with decoupled weight decay regularization
- `CrossEntropyLoss` - standard loss function for multi-class classification
- `CosineAnnealingLR` scheduler - gradually reduces learning rate following a cosine curve, helping the model converge
- Early stopping - halts training when validation loss stops improving (patience = 10 epochs), preventing overfitting
- Best model checkpointing - saves the weights from the epoch with the lowest validation loss

**Evaluation**
- Separate evaluation on validation set (during training) and held-out test set (final accuracy)
- Accuracy tracking across epochs via `train_acc` / `val_acc`

**Hardware Acceleration**
- Automatic device selection: CUDA (NVIDIA GPU) → MPS (Apple Silicon) → CPU fallback
