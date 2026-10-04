# Task 1 - Experiment Results

## Dataset and Preprocessing

- Dataset: CIFAR-10
- Training images: 50,000
- Test images: 10,000
- Image size: 32 × 32 × 3
- Number of classes: 10
- Training-validation split: 45,000 training images and 5,000 validation images
- Random seed: 42
- Pixel normalization: Pixel values were scaled from 0–255 to 0–1 by dividing by 255.0

## Baseline CNN

| Metric | Result |

|---|---:|

| Train Accuracy | 79.94% |

| Validation Accuracy | 67.02% |

| Test Accuracy | 67.55% |

| Test Loss | 1.0607 |

| Parameters | 315,722 |

| Epochs | 10 |

| Batch Size | 64 |

| Optimizer | Adam |

### Observation

The baseline CNN achieved 67.55% test accuracy. The train-validation gap was 12.92 percentage points, indicating some overfitting.

---

## Dropout CNN

| Metric | Result |

|---|---:|

| Train Accuracy | 70.17% |

| Validation Accuracy | 69.40% |

| Test Accuracy | 69.86% |

| Test Loss | 0.8761 |

| Parameters | 315,722 |

| Dropout Rate | 0.5 |

| Epochs | 10 |

| Batch Size | 64 |

| Optimizer | Adam |

### Observation

Adding Dropout reduced the training accuracy but improved validation and test accuracy. The train-validation gap reduced to 0.77 percentage points, showing better generalization.

### Improvement over Baseline

Test accuracy improved from 67.55% to 69.86%, an improvement of 2.31 percentage points.

---

## L2 Regularization CNN

| Metric | Result |

|---|---:|

| Train Accuracy | 74.95% |

| Validation Accuracy | 67.54% |

| Test Accuracy | 66.79% |

| Test Loss | 1.1574 |

| Parameters | 315,722 |

| L2 Regularization | 0.001 |

| Epochs | 10 |

| Batch Size | 64 |

| Optimizer | Adam |

### Observation

L2 Regularization reduced the training accuracy compared with the baseline, but it did not improve validation or test accuracy. The test accuracy decreased from 67.55% (baseline) to 66.79%.

### Improvement over Baseline

Test accuracy changed from 67.55% to 66.79%, a decrease of 0.76 percentage points.

---

## Dropout + L2 Regularization CNN

| Metric | Result |

|---|---:|

| Train Accuracy | 81.61% |

| Validation Accuracy | 67.84% |

| Test Accuracy | 68.44% |

| Test Loss | 1.0618 |

| Parameters | 315,722 |

| Dropout Rate | 0.5 |

| L2 Regularization | 0.001 |

| Epochs | 10 |

| Batch Size | 64 |

| Optimizer | Adam |

### Observation

Combining Dropout and L2 Regularization increased the test accuracy to 68.44%. However, the train-validation gap was 13.77 percentage points, indicating that overfitting was still present.

### Improvement over Baseline

Test accuracy improved from 67.55% to 68.44%, an improvement of 0.89 percentage points.

---

## Horizontal Flip CNN

| Metric | Result |

|---|---:|

| Train Accuracy | 76.84% |

| Validation Accuracy | 70.82% |

| Test Accuracy | 70.14% |

| Test Loss | 0.8882 |

| Parameters | 315,722 |

| Horizontal Flip | Enabled |

| Epochs | 10 |

| Batch Size | 64 |

| Optimizer | Adam |

### Observation

Adding horizontal flip augmentation improved the test accuracy from 67.55% to 70.14%. The train-validation gap also decreased from 12.92 to 6.02 percentage points, indicating better generalization.

### Improvement over Baseline

Test accuracy improved from 67.55% to 70.14%, an improvement of 2.59 percentage points.

---

## Rotation CNN

| Metric | Result |

|---|---:|

| Train Accuracy | 67.67% |

| Validation Accuracy | 68.40% |

| Test Accuracy | 67.55% |

| Test Loss | 0.9437 |

| Parameters | 315,722 |

| Rotation | 0.1 |

| Epochs | 10 |

| Batch Size | 64 |

| Optimizer | Adam |

### Observation

Adding rotation augmentation did not improve the test accuracy compared with the baseline. The baseline and rotation model both achieved 67.55% test accuracy. Therefore, rotation did not provide a measurable improvement in this experiment.

### Improvement over Baseline

Test accuracy changed from 67.55% to 67.55%, resulting in 0.00 percentage points improvement.

---

## Zoom Augmentation CNN

| Metric | Result |

|---|---:|

| Train Accuracy | 77.85% |

| Validation Accuracy | 67.06% |

| Test Accuracy | 67.15% |

| Test Loss | 1.0372 |

| Parameters | 315,722 |

| Zoom | 0.1 |

| Epochs | 10 |

| Batch Size | 64 |

| Optimizer | Adam |

### Observation

Adding Zoom augmentation did not improve the test accuracy compared with the baseline. The test accuracy decreased slightly from 67.55% to 67.15%. The train-validation gap was 10.79 percentage points, which was lower than the baseline gap of 12.92 percentage points.

### Improvement over Baseline

Test accuracy changed from 67.55% to 67.15%, resulting in a decrease of 0.40 percentage points.

---

## Brightness Augmentation CNN

| Metric | Result |

|---|---:|

| Train Accuracy | 79.32% |

| Validation Accuracy | 68.18% |

| Test Accuracy | 67.78% |

| Test Loss | 1.0232 |

| Parameters | 315,722 |

| Brightness | 0.1 |

| Epochs | 10 |

| Batch Size | 64 |

| Optimizer | Adam |

### Observation

Adding brightness augmentation produced only a small improvement in test accuracy compared with the baseline. The test accuracy increased from 67.55% to 67.78%. The train-validation gap was 11.14 percentage points.

### Improvement over Baseline

Test accuracy improved from 67.55% to 67.78%, an improvement of 0.23 percentage points.

### Conclusion

Brightness augmentation did not provide a meaningful improvement in this experiment.

## SGD Optimization CNN

| Metric | Result |

|---|---:|

| Train Accuracy | 58.49% |

| Validation Accuracy | 47.74% |

| Test Accuracy | 47.90% |

| Test Loss | 1.4762 |

| Parameters | 315,722 |

| Optimizer | SGD |

| Epochs | 10 |

| Batch Size | 64 |

| Training Time | 194 seconds |

### Observation

Replacing Adam with SGD resulted in a significant decrease in performance. The model achieved only 47.90% test accuracy compared with 67.55% for the baseline Adam model. The model also learned more slowly during the 10 epochs.

### Improvement over Baseline

Test accuracy changed from 67.55% to 47.90%, resulting in a decrease of 19.65 percentage points.

---

## RMSprop Optimization CNN

| Metric | Result |

|---|---:|

| Train Accuracy | 87.02% |

| Validation Accuracy | 68.64% |

| Test Accuracy | 68.28% |

| Test Loss | 1.1414 |

| Parameters | 315,722 |

| Optimizer | RMSprop |

| Epochs | 10 |

| Batch Size | 64 |

### Observation

RMSprop slightly improved the test accuracy compared with the baseline. However, the train-validation gap increased to 18.38 percentage points, indicating considerable overfitting.

### Improvement over Baseline

Test accuracy improved from 67.55% to 68.28%, an improvement of 0.73 percentage points.

---

## Adam with Lower Learning Rate CNN

| Metric | Result |

|---|---:|

| Train Accuracy | 62.23% |

| Validation Accuracy | 62.16% |

| Test Accuracy | 61.48% |

| Test Loss | 1.1072 |

| Parameters | 315,722 |

| Learning Rate | 0.0001 |

| Optimizer | Adam |

| Epochs | 10 |

| Batch Size | 64 |

| Training Time | 157 seconds |

### Observation

Using a lower learning rate of 0.0001 with Adam greatly reduced the train-validation gap to only 0.07 percentage points. However, the model underfit during the 10 epochs and achieved lower test accuracy than the baseline.

### Improvement over Baseline

Test accuracy changed from 67.55% to 61.48%, resulting in a decrease of 6.07 percentage points.

---

# Group 3 Conclusion — Optimization

Three optimization settings were evaluated: SGD, RMSprop, and Adam with a lower learning rate.

- SGD performed significantly worse than the baseline, with a decrease of 19.65 percentage points.

- RMSprop slightly improved test accuracy by 0.73 percentage points, but showed a high train-validation gap of 18.38 percentage points.

- Adam with learning rate 0.0001 produced a very small train-validation gap of 0.07 percentage points, but the test accuracy decreased by 6.07 percentage points, indicating underfitting within 10 epochs.

- The baseline Adam configuration achieved better overall performance than the tested optimization alternatives.

### Group 3 Winner

**Baseline Adam configuration** was selected as the best optimization setting among the tested configurations.

The optimization experiments demonstrate that optimizer choice and learning rate can significantly affect CNN training and generalization.

# MASTER EXPERIMENT TABLE

| Group | Experiment | Main Change | Train Acc | Val Acc | Test Acc | Test Loss | Train-Val Gap | Parameters | Training Time | vs Baseline |

|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|

| Baseline | Baseline CNN | None | 79.94% | 67.02% | 67.55% | 1.0607 | 12.92 pp | 315,722 | ~3–4 min | — |

| Regularization | Dropout | Dropout 0.5 | 70.17% | 69.40% | 69.86% | 0.8761 | 0.77 pp | 315,722 | Not recorded | +2.31 pp |

| Regularization | L2 | L2 0.001 | 74.95% | 67.54% | 66.79% | 1.1574 | 7.41 pp | 315,722 | Not recorded | -0.76 pp |

| Regularization | Dropout + L2 | L2 0.001 + Dropout 0.5 | 81.61% | 67.84% | 68.44% | 1.0618 | 13.77 pp | 315,722 | Not recorded | +0.89 pp |

| Augmentation | Horizontal Flip | RandomFlip horizontal | 76.84% | 70.82% | 70.14% | 0.8882 | 6.02 pp | 315,722 | Not recorded | +2.59 pp |

| Augmentation | Rotation | RandomRotation 0.1 | 67.67% | 68.40% | 67.55% | 0.9437 | -0.73 pp | 315,722 | Not recorded | 0.00 pp |

| Augmentation | Zoom | RandomZoom 0.1 | 77.85% | 67.06% | 67.15% | 1.0372 | 10.79 pp | 315,722 | Not recorded | -0.40 pp |

| Augmentation | Brightness | RandomBrightness 0.1 | 79.32% | 68.18% | 67.78% | 1.0232 | 11.14 pp | 315,722 | Not recorded | +0.23 pp |

| Optimization | SGD | Optimizer: SGD | 58.49% | 47.74% | 47.90% | 1.4762 | 10.75 pp | 315,722 | 194 sec | -19.65 pp |

| Optimization | RMSprop | Optimizer: RMSprop | 87.02% | 68.64% | 68.28% | 1.1414 | 18.38 pp | 315,722 | Not recorded | +0.73 pp |

| Optimization | Adam Low LR | Adam, learning rate 0.0001 | 62.23% | 62.16% | 61.48% | 1.1072 | 0.07 pp | 315,722 | 157 sec | -6.07 pp |

| Architecture | Depth Change | Additional Conv2D 128 + MaxPooling2D | 79.46% | 72.66% | 70.74% | 0.8894 | 6.80 pp | 160,202 | 171.62 sec | +3.19 pp |

| Architecture | Kernel Size | First Conv2D changed to 5x5 | 81.03% | 65.32% | 64.77% | 1.1779 | 15.71 pp | 317,258 | 130.59 sec | -2.78 pp |

## Overall Experiment Summary

### Best Results by Group

- **Baseline:** Test Accuracy = 67.55%

- **Regularization Winner:** Dropout — Test Accuracy = 69.86%

- **Augmentation Winner:** Horizontal Flip — Test Accuracy = 70.14%

- **Optimization Winner:** Baseline Adam configuration — Test Accuracy = 67.55%
### Current Best Final Model

### Current Best Final Model

The final customized CNN achieved a test accuracy of **70.10%**. It combines the techniques selected from the experiments: increased depth and horizontal flip augmentation, with Dropout 0.5 for regularization.

Compared with the baseline CNN, the final model improved test accuracy from 67.55% to 70.10%, an improvement of **2.55 percentage points**. The final model also achieved a train-validation gap of 2.31 percentage points, indicating a relatively small generalization gap in this run.

The standalone Depth Change experiment achieved 70.74% test accuracy, while the final customized CNN combines multiple selected techniques.
---

## Depth Change CNN

| Metric | Result |

|---|---:|

| Train Accuracy | 79.46% |

| Validation Accuracy | 72.66% |

| Test Accuracy | 70.74% |

| Test Loss | 0.8894 |

| Parameters | 160,202 |

| Additional Conv Block | Conv2D 128 filters + MaxPooling2D |

| Epochs | 10 |

| Batch Size | 64 |

| Optimizer | Adam |

| Training Time | 171.62 seconds |

### Observation

An additional convolutional block with 128 filters was added to increase the depth of the CNN. The deeper model achieved 70.74% test accuracy compared with 67.55% for the baseline.

The validation accuracy also increased from 67.02% to 72.66%, while the train-validation gap decreased from 12.92 to 6.80 percentage points. This indicates better generalization in this experiment.

The parameter count decreased from 315,722 to 160,202 because the additional pooling layer reduced the feature-map size before the Flatten layer.

### Improvement over Baseline

Test accuracy improved from 67.55% to 70.74%, an improvement of **3.19 percentage points**.

### Architecture Trade-off

The deeper architecture provided better test and validation accuracy while using fewer parameters in this particular design. Training time was approximately 171.62 seconds.

### Learning

Increasing CNN depth can improve feature extraction and generalization, but the effect depends on how pooling and feature-map sizes are designed. Adding a convolutional block does not necessarily increase the total parameter count.

## Kernel Size Change CNN

### Experiment

The kernel size of the first convolutional layer was changed from 3x3 to 5x5.

### Configuration

- First Conv2D: 32 filters, 5x5 kernel

- Second Conv2D: 64 filters, 3x3 kernel

- Optimizer: Adam

- Epochs: 10

- Batch Size: 64

- Dataset Split: 45,000 training / 5,000 validation / 10,000 test

### Results

- Train Accuracy: 81.03%

- Validation Accuracy: 65.32%

- Test Accuracy: 64.77%

- Test Loss: 1.1779

- Train-Validation Gap: 15.71 percentage points

- Parameters: 317,258

- Training Time: 130.59 seconds

### Comparison with Baseline

- Baseline Test Accuracy: 67.55%

- Kernel Size Test Accuracy: 64.77%

- Change: -2.78 percentage points

### Observation

Changing the first convolutional kernel from 3x3 to 5x5 did not improve the model. The test accuracy decreased from 67.55% to 64.77%.

The larger kernel also increased the number of parameters from 315,722 to 317,258.

The train-validation gap increased to 15.71 percentage points, indicating higher overfitting compared with the baseline.

### Conclusion

The 5x5 kernel was not selected for the final model because it produced lower test accuracy and a larger train-validation gap than the baseline.

### Learning

Kernel size affects the features captured by convolutional layers and also affects the number of parameters. A larger kernel is not necessarily better; the result depends on the dataset and model architecture.

## Group 4 Architecture — Conclusion

### Architecture Experiments Compared

| Experiment | Test Accuracy | Parameters | Training Time | vs Baseline |

|---|---:|---:|---:|---:|

| Depth Change | 70.74% | 160,202 | 171.62 sec | +3.19 pp |

| Kernel Size 5x5 | 64.77% | 317,258 | 130.59 sec | -2.78 pp |

### Group Winner

The **Depth Change** experiment performed better than the Kernel Size Change experiment.

The Depth Change model achieved **70.74% test accuracy**, which was **3.19 percentage points higher than the baseline**.

The 5x5 Kernel Size experiment achieved only **64.77% test accuracy**, which was **2.78 percentage points lower than the baseline**.

Therefore, **Depth Change is selected as the architecture technique for the final model**.

### Learning

Increasing model depth can improve feature extraction and performance, but changing kernel size does not necessarily improve accuracy. The effect depends on the overall architecture and dataset.

from tensorflow.keras import layers, models

from tensorflow.keras.datasets import cifar10

Final Customized CNN

The final customized CNN combines the techniques selected from the completed experiments: increased CNN depth, horizontal flip augmentation, and Dropout regularization. The optimization setting remains the baseline Adam configuration because the tested alternatives did not improve overall performance.

Final Model Results

Metric

Result

Train Accuracy

71.30%

Validation Accuracy

71.02%

Test Accuracy

70.49%

Test Loss

0.8594

Train-Validation Gap

0.28 percentage points

Parameters

160,202

Training Time

154.74 seconds

Epochs

10

Batch Size

64

Optimizer

Adam

Comparison with Baseline

Baseline Test Accuracy: 67.55%

Final Test Accuracy: 70.49%

Improvement: 2.94 percentage points

Baseline Parameters: 315,722

Final Parameters: 160,202

Final Training Time: 154.74 seconds

Confusion Matrix Analysis

The final model generated a confusion matrix saved as final_confusion_matrix.png. The most frequent misclassifications in this run were:

Actual Class

Predicted Class

Count

Dog

Cat

221

Automobile

Truck

160

Cat

Dog

154

Bird

Cat

127

Deer

Horse

105

The model had the most difficulty distinguishing visually similar classes, especially dog and cat.

Final Conclusion

The final customized CNN achieved 70.49% test accuracy, improving over the 67.55% baseline by 2.94 percentage points. The final model also had a small train-validation gap of 0.28 percentage points. The improvement is close to the target range, but the actual measured result is reported without modification.

The final model artifacts include the accuracy curve, loss curve, and confusion matrix.