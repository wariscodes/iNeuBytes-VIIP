# Task 2 — Sentiment Analysis Experiment Results

## Baseline Experiment

### TF-IDF + Logistic Regression

**Experiment Type:** Baseline ML Model

**Text Representation:**
- TF-IDF
- Unigram only (`ngram_range=(1,1)`)
- Vocabulary/features: 111,308

**Dataset:**
- Training reviews: 25,000
- Testing reviews: 25,000
- Positive and negative reviews are balanced.

**Results:**

| Metric | Score |
|---|---:|
| Accuracy | 88.52% |
| Precision | 88.43% |
| Recall | 88.63% |
| F1 Score | 88.53% |
| Training Time | 1.07 seconds |

### Conclusion

The TF-IDF + Logistic Regression model achieved 88.52% test accuracy with balanced Precision, Recall, and F1 scores. This result will be used as the baseline for comparison with SVM, TF-IDF experiments, and the LSTM model.

---

## SVM Experiment

### TF-IDF + Linear SVM

**Experiment Type:** ML Model Comparison

**Text Representation:**
- TF-IDF
- Unigram only (`ngram_range=(1,1)`)
- Vocabulary/features: 111,308

**Dataset:**
- Training reviews: 25,000
- Testing reviews: 25,000
- Same train/test split used for Logistic Regression comparison.

**Results:**

| Metric | Score |
|---|---:|
| Accuracy | 88.17% |
| Precision | 88.70% |
| Recall | 87.48% |
| F1 Score | 88.09% |
| Training Time | 0.47 seconds |

### Comparison with Logistic Regression

Logistic Regression achieved slightly higher Accuracy and F1 Score, while Linear SVM achieved higher Precision and lower training time. Both models performed similarly on the IMDb test set.

---

## TF-IDF Bigram Experiment

### TF-IDF Unigram + Bigram + Logistic Regression

**Experiment Type:** TF-IDF Feature Experiment

**Main Variable Changed:**
- Baseline: Unigram (`ngram_range=(1,1)`)
- Experiment: Unigram + Bigram (`ngram_range=(1,2)`)

**Dataset:**
- Training reviews: 25,000
- Testing reviews: 25,000
- Same train/test split and preprocessing used.

**Results:**

| Metric | Score |
|---|---:|
| Accuracy | 88.68% |
| Precision | 87.94% |
| Recall | 89.64% |
| F1 Score | 88.78% |
| TF-IDF Features | 1,616,416 |
| Training Time | 6.43 seconds |

### Comparison with Unigram Baseline

The bigram configuration produced a small improvement in Accuracy and F1 Score, while Recall improved more noticeably. However, the number of TF-IDF features increased substantially from 111,308 to 1,616,416, along with higher training time.

This experiment shows that adding word-pair information can improve sentiment classification slightly, but it also increases the feature space considerably.

---

## TF-IDF Max Features Experiment

### TF-IDF with 50,000 Features + Logistic Regression

**Experiment Type:** TF-IDF Vocabulary Size Experiment

**Main Variable Changed:**
- Baseline: No `max_features` limit
- Experiment: `max_features=50000`

**Dataset:**
- Training reviews: 25,000
- Testing reviews: 25,000
- Same train/test split and preprocessing used.

**Results:**

| Metric | Score |
|---|---:|
| Accuracy | 88.51% |
| Precision | 88.48% |
| Recall | 88.56% |
| F1 Score | 88.52% |
| TF-IDF Features | 50,000 |
| Training Time | 0.85 seconds |

### Comparison with Unigram Baseline

Limiting the TF-IDF vocabulary to 50,000 features produced almost the same performance as the unrestricted unigram baseline. The feature count decreased substantially, while training time also decreased slightly.

This experiment shows that reducing the vocabulary can preserve similar classification performance while using fewer features.

---

## LSTM Baseline Experiment

### LSTM Sentiment Classification

**Experiment Type:** Deep Learning Baseline

**Text Representation:**
- Tokenization using Keras Tokenizer
- Maximum vocabulary size: 20,000
- Sequence length: 200
- Padding: post-padding
- Truncation: post-truncation

**Model Architecture:**
- Embedding layer: 128 dimensions
- LSTM: 64 units
- Dropout: 0.2
- Output layer: 1 neuron with sigmoid activation

**Training Configuration:**
- Optimizer: Adam
- Loss: Binary Crossentropy
- Epochs: 5
- Batch size: 64
- Random seed: 42

**Dataset:**
- Training reviews: 20,000
- Validation reviews: 5,000
- Testing reviews: 25,000
- Positive and negative reviews are balanced.

**Model Parameters:**
- Total parameters: 2,609,473

**Results:**

| Metric | Score |
|---|---:|
| Train Accuracy | 95.74% |
| Validation Accuracy | 67.42% |
| Test Accuracy | 80.36% |
| Precision | 74.08% |
| Recall | 93.42% |
| F1 Score | 82.63% |
| Train-Validation Gap | 28.32 percentage points |
| Training Time | 237.76 seconds |

### Observation

The LSTM achieved 80.36% test accuracy with an F1 Score of 82.63%. Recall was relatively high at 93.42%, while Precision was 74.08%.

The model showed significant overfitting. Training accuracy reached 95.74%, while validation accuracy decreased to 67.42% by the fifth epoch, producing a train-validation gap of 28.32 percentage points.

The validation accuracy was 80.98% in the first epoch but decreased in later epochs while training accuracy continued to increase. This indicates that the model started memorizing the training data rather than improving its generalization.

### Learning

The baseline LSTM provides a starting point for controlled experiments. The large train-validation gap shows that changes such as dropout, sequence length, architecture, or regularization should be investigated to reduce overfitting and improve generalization.

---

## LSTM Dropout Experiment

### LSTM with Dropout 0.5

**Experiment Type:** LSTM Regularization Experiment

**Main Variable Changed:**
- Baseline Dropout: 0.2
- Experiment Dropout: 0.5

**All Other Configuration Kept Same:**
- Vocabulary size: 20,000
- Sequence length: 200
- Embedding dimension: 128
- LSTM units: 64
- Optimizer: Adam
- Epochs: 5
- Batch size: 64
- Random seed: 42

**Model Parameters:**
- Total parameters: 2,609,473

**Results:**

| Metric | Score |
|---|---:|
| Train Accuracy | 94.17% |
| Validation Accuracy | 57.62% |
| Test Accuracy | 76.87% |
| Precision | 69.39% |
| Recall | 96.15% |
| F1 Score | 80.61% |
| Train-Validation Gap | 36.55 percentage points |
| Training Time | 243.07 seconds |

### Comparison with LSTM Baseline

| Metric | Baseline LSTM | Dropout 0.5 |
|---|---:|---:|
| Test Accuracy | 80.36% | 76.87% |
| Precision | 74.08% | 69.39% |
| Recall | 93.42% | 96.15% |
| F1 Score | 82.63% | 80.61% |
| Train-Validation Gap | 28.32 pp | 36.55 pp |

### Conclusion

Increasing Dropout from 0.2 to 0.5 did not improve the overall LSTM performance in this experiment. Test Accuracy decreased from 80.36% to 76.87%, while F1 Score decreased from 82.63% to 80.61%.

Although Recall increased from 93.42% to 96.15%, the model had lower Precision and lower overall test performance.

The train-validation gap also increased from 28.32 percentage points to 36.55 percentage points. Therefore, Dropout 0.5 is not selected as an improvement over the baseline configuration.

The validation accuracy reached 86.98% in the first epoch but decreased substantially in later epochs, indicating that the model's generalization deteriorated during continued training.

---

## LSTM Sequence Length Experiment

### LSTM with Sequence Length 100

**Experiment Type:** LSTM Sequence Length Experiment

**Main Variable Changed:**
- Baseline Sequence Length: 200
- Experiment Sequence Length: 100

**All Other Configuration Kept Same:**
- Vocabulary size: 20,000
- Embedding dimension: 128
- LSTM units: 64
- Dropout: 0.2
- Optimizer: Adam
- Epochs: 5
- Batch size: 64
- Random seed: 42

**Model Parameters:**
- Total parameters: 2,609,473

**Results:**

| Metric | Score |
|---|---:|
| Train Accuracy | 97.65% |
| Validation Accuracy | 81.36% |
| Test Accuracy | 78.05% |
| Precision | 81.03% |
| Recall | 73.24% |
| F1 Score | 76.94% |
| Train-Validation Gap | 16.29 percentage points |
| Training Time | 126.58 seconds |

### Comparison with LSTM Baseline

| Metric | Baseline LSTM | Sequence Length 100 |
|---|---:|---:|
| Test Accuracy | 80.36% | 78.05% |
| Precision | 74.08% | 81.03% |
| Recall | 93.42% | 73.24% |
| F1 Score | 82.63% | 76.94% |
| Train-Validation Gap | 28.32 pp | 16.29 pp |
| Training Time | 237.76 sec | 126.58 sec |

### Conclusion

Reducing the sequence length from 200 to 100 reduced the training time from 237.76 seconds to 126.58 seconds and reduced the train-validation gap from 28.32 to 16.29 percentage points.

However, Test Accuracy decreased from 80.36% to 78.05% and F1 Score decreased from 82.63% to 76.94%. Recall also decreased substantially, although Precision increased.

Therefore, Sequence Length 100 is not selected as an overall improvement over the baseline LSTM. The experiment shows that a shorter sequence can reduce training time and overfitting, but it may also remove useful contextual information required for sentiment classification.

## LSTM Embedding Dimension Experiment

### Experiment Goal

The embedding dimension was reduced from 128 to 64 to study its effect on model performance, model size, overfitting, and training time.

Only the embedding dimension was changed. All other parameters were kept the same as the LSTM baseline.

### Configuration

- Vocabulary Size: 20,000
- Sequence Length: 200
- Embedding Dimension: 64
- LSTM Units: 64
- Dropout: 0.2
- Optimizer: Adam
- Epochs: 5
- Batch Size: 64
- Random Seed: 42
- Training Samples: 20,000
- Validation Samples: 5,000
- Test Samples: 25,000

### Results

| Metric | LSTM Baseline (Embedding 128) | Embedding 64 |
|---|---:|---:|
| Parameters | 2,609,473 | 1,313,089 |
| Train Accuracy | 95.74% | 90.38% |
| Validation Accuracy | 67.42% | 64.36% |
| Test Accuracy | 80.36% | 78.72% |
| Precision | 74.08% | 72.48% |
| Recall | 93.42% | 92.62% |
| F1 Score | 82.63% | 81.32% |
| Train-Val Gap | 28.32 pp | 26.02 pp |
| Training Time | 237.76 sec | 188.35 sec |

### Comparison

Reducing the embedding dimension from 128 to 64 reduced the number of model parameters from 2,609,473 to 1,313,089.

The training time also decreased from 237.76 seconds to 188.35 seconds.

However, test accuracy decreased from 80.36% to 78.72%, while F1 score decreased from 82.63% to 81.32%.

The train-validation gap also decreased from 28.32 percentage points to 26.02 percentage points, indicating a small reduction in overfitting.

### Conclusion

The embedding dimension of 64 significantly reduced model size and training time, but it also reduced test accuracy and F1 score compared with the baseline embedding dimension of 128.

Therefore, embedding dimension 64 did not provide an overall performance improvement. The reduction in model size and training time is useful as a trade-off, but embedding dimension 128 remains preferable for the current LSTM configuration based on the tested results.

## LSTM Units Experiment

### Experiment Goal

The number of LSTM units was increased from 64 to 128 to study the effect of model capacity on sentiment classification performance, model size, overfitting, and training time.

Only the number of LSTM units was changed. All other parameters were kept the same as the LSTM baseline.

### Configuration

- Vocabulary Size: 20,000
- Sequence Length: 200
- Embedding Dimension: 128
- LSTM Units: 128
- Dropout: 0.2
- Optimizer: Adam
- Epochs: 5
- Batch Size: 64
- Random Seed: 42
- Training Samples: 20,000
- Validation Samples: 5,000
- Test Samples: 25,000

### Results

| Metric | LSTM Baseline (64 Units) | LSTM 128 Units |
|---|---:|---:|
| Parameters | 2,609,473 | 2,691,713 |
| Train Accuracy | 95.74% | 88.73% |
| Validation Accuracy | 67.42% | 67.94% |
| Test Accuracy | 80.36% | 74.00% |
| Precision | 74.08% | 71.99% |
| Recall | 93.42% | 78.58% |
| F1 Score | 82.63% | 75.14% |
| Train-Val Gap | 28.32 pp | 20.79 pp |
| Training Time | 237.76 sec | 359.14 sec |

### Comparison

Increasing the number of LSTM units from 64 to 128 increased the model parameters from 2,609,473 to 2,691,713.

Training time also increased from 237.76 seconds to 359.14 seconds.

However, test accuracy decreased from 80.36% to 74.00%, and F1 score decreased from 82.63% to 75.14%.

The train-validation gap decreased from 28.32 percentage points to 20.79 percentage points, indicating a smaller train-validation gap, but this did not translate into better test performance.

### Conclusion

Increasing the LSTM units from 64 to 128 did not improve the overall sentiment classification performance in this experiment.

The 128-unit model required more training time and produced lower test accuracy, precision, recall, and F1 score than the 64-unit baseline.

Therefore, the 128-unit configuration was not selected for the final LSTM model based on the tested results.

## LSTM Experiment Comparison and Final Configuration

### Comparison of LSTM Experiments

| Experiment | Test Accuracy | Precision | Recall | F1 Score | Parameters | Training Time |
|---|---:|---:|---:|---:|---:|---:|
| LSTM Baseline — Embedding 128 | 80.36% | 74.08% | 93.42% | 82.63% | 2,609,473 | 237.76 sec |
| Dropout 0.5 | 76.87% | 69.39% | 96.15% | 80.61% | 2,609,473 | 243.07 sec |
| Sequence Length 100 | 78.05% | 81.03% | 73.24% | 76.94% | 2,609,473 | 126.58 sec |
| Embedding Dimension 64 | 78.72% | 72.48% | 92.62% | 81.32% | 1,313,089 | 188.35 sec |
| LSTM Units 128 | 74.00% | 71.99% | 78.58% | 75.14% | 2,691,713 | 359.14 sec |

### Final LSTM Configuration

Based on the tested experiments, the baseline LSTM configuration was retained as the final LSTM configuration because it achieved the highest test accuracy and F1 score among the tested LSTM variants.

- Vocabulary Size: 20,000
- Sequence Length: 200
- Embedding Dimension: 128
- LSTM Units: 64
- Dropout: 0.2
- Optimizer: Adam
- Epochs: 5
- Batch Size: 64
- Test Accuracy: 80.36%
- Precision: 74.08%
- Recall: 93.42%
- F1 Score: 82.63%

The experiments also showed useful trade-offs. Reducing the embedding dimension decreased model size and training time, while reducing sequence length reduced training time and the train-validation gap. However, these configurations did not improve the overall test performance. Increasing dropout or LSTM units also did not improve the overall results.

## Final Analysis Outputs

A final analysis run was performed to generate the required final artifacts for Task 2. The analysis used the same 25,000-review IMDb test set for Logistic Regression, Linear SVM, and the final LSTM model.

### Final Analysis LSTM Configuration

- Vocabulary size: 20,000
- Sequence length: 200
- Embedding dimension: 128
- LSTM units: 64
- Dropout: 0.2
- Optimizer: Adam
- Epochs: 5
- Batch size: 64
- Random seed: 42
- Training samples: 20,000
- Validation samples: 5,000
- Test samples: 25,000
- Model parameters: 2,609,473

### Final Analysis LSTM Results

- Accuracy: 78.62%
- Precision: 84.77%
- Recall: 69.78%
- F1-score: 76.55%

### Generated Final Artifacts

The following files were successfully generated and verified:

1. `confusion_matrix_logistic_regression.png`
2. `confusion_matrix_linear_svm.png`
3. `confusion_matrix_final_lstm.png`
4. `lstm_training_accuracy.png`
5. `lstm_training_loss.png`
6. `misclassified_examples.txt`

The misclassified examples file contains 5,344 incorrectly classified test reviews for further error analysis.

### Note

The final analysis LSTM run was used to generate the final analysis artifacts. Its metrics are recorded separately from the previously documented LSTM baseline experiment because the final analysis used a separately stratified training/validation split.

## Final ML vs LSTM Comparison

The final analysis evaluated Logistic Regression, Linear SVM, and a separately rerun final LSTM model using the same 25,000-review IMDb test set. The LSTM rerun used a separately stratified training/validation split and was used to generate the final analysis artifacts.

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 88.52% | 88.43% | 88.63% | 88.53% |
| Linear SVM | 88.17% | 88.70% | 87.48% | 88.09% |
| Final LSTM Analysis Run | 78.62% | 84.77% | 69.78% | 76.55% |

### Comparison Observations

- Logistic Regression achieved 88.52% accuracy and 88.53% F1-score.
- Linear SVM achieved 88.17% accuracy and 88.09% F1-score.
- The final LSTM analysis run achieved 78.62% accuracy and 76.55% F1-score.
- Logistic Regression and Linear SVM used TF-IDF text features, while the LSTM used tokenized and padded sequences with an embedding layer and recurrent layer.
- In this implementation, the final LSTM analysis run did not outperform the TF-IDF-based classical ML models.
- The confusion matrices, LSTM training curves, and misclassified examples provide additional evidence for model evaluation and error analysis.
- The previously recorded LSTM baseline experiment remains documented separately with its original metrics and was not overwritten by the final analysis rerun.

## Task 2 Master Experiment Table

| Experiment | Main Change | Accuracy | Precision | Recall | F1 Score | Train-Val Gap | Parameters | Training Time | Result |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| TF-IDF + Logistic Regression | Baseline unigram TF-IDF | 88.52% | 88.43% | 88.63% | 88.53% | N/A | N/A | 1.07 sec | Baseline ML |
| TF-IDF + Linear SVM | Linear SVM with unigram TF-IDF | 88.17% | 88.70% | 87.48% | 88.09% | N/A | N/A | 0.47 sec | ML comparison |
| TF-IDF Bigram | Unigram → Unigram + Bigram | 88.68% | 87.94% | 89.64% | 88.78% | N/A | N/A | 6.43 sec | Small performance improvement, much larger feature space |
| TF-IDF Max Features | Unlimited → 50,000 features | 88.51% | 88.48% | 88.56% | 88.52% | N/A | N/A | 0.85 sec | Similar performance with fewer features |
| LSTM Baseline | Embedding 128, Sequence 200, Units 64, Dropout 0.2 | 80.36% | 74.08% | 93.42% | 82.63% | 28.32 pp | 2,609,473 | 237.76 sec | LSTM baseline |
| LSTM Dropout | Dropout 0.2 → 0.5 | 76.87% | 69.39% | 96.15% | 80.61% | 36.55 pp | 2,609,473 | 243.07 sec | Not selected |
| LSTM Sequence Length | Sequence 200 → 100 | 78.05% | 81.03% | 73.24% | 76.94% | 16.29 pp | 2,609,473 | 126.58 sec | Faster, but lower overall performance |
| LSTM Embedding | Embedding 128 → 64 | 78.72% | 72.48% | 92.62% | 81.32% | 26.02 pp | 1,313,089 | 188.35 sec | Smaller/faster, but lower performance |
| LSTM Units | LSTM units 64 → 128 | 74.00% | 71.99% | 78.58% | 75.14% | 20.79 pp | 2,691,713 | 359.14 sec | Not selected |
| Final LSTM Analysis Run | Final analysis rerun with stratified split | 78.62% | 84.77% | 69.78% | 76.55% | 6.12 pp | 2,609,473 | Not separately recorded | Used for final analysis artifacts |

### Master Table Note

The Final LSTM Analysis Run is documented separately from the original LSTM baseline experiment because it used a separately stratified training/validation split. Its metrics were used for the final analysis artifacts and ML vs LSTM comparison.