# AI354 Lab Assignment 4: Comparative Analysis of Neural Architectures - CNNs for Vision and BiLSTMs for NLP

## Devesh Singh Chauhan
### I23MA002

**Assigned on:** 29/01/26

---

## Datasets

### Q1. Image Classification Dataset
- **Clothes Dataset**: [Link](https://www.kaggle.com/datasets/ryanbadai/clothes-dataset)
- Same dataset as previous assignment (balanced classes)

### Q2. Sentiment Analysis Dataset
- **Amazon Reviews Dataset**: [Link](https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews/data)
- Binary sentiment classification (positive/negative)

---

## Libraries Allowed
- numpy
- pandas
- sklearn
- pytorch
- matplotlib
- seaborn
- nltk (for text preprocessing)
- (other relevant Python libraries)

---

## Assignment Questions

### Q1. Convolutional Neural Network (CNN) for Image Classification

Use the same dataset given in the previous assignment (clothes dataset).

**Tasks:**
- Design and train a **Convolutional Neural Network (CNN)** to classify clothing images into their respective categories using PyTorch
- Compare your results with the previous assignment model performance (Logistic Regression/MLP)
- Show how the model performance is impacted after **decreasing the number of convolution layers** in your implementation

**Implementation Requirements:**
- CNN architecture design with appropriate layers (convolution, pooling, fully connected)
- Image preprocessing and augmentation (if applicable)
- Training pipeline with appropriate loss function and optimizer
- Performance comparison table with previous models
- Ablation study showing impact of reducing convolutional layers

---

### Q2. Bidirectional LSTM (BiLSTM) for Sentiment Classification

Use the Amazon review dataset for sentiment classification (positive/negative).

#### Part A: Model Training with Different Dropout Values
Train a **BiLSTM model** with the following split:
- **80%** Training data
- **10%** Validation data
- **10%** Test data

**Experiment with different dropout values at LSTM and embedding layers:**
- Dropout = 0.2
- Dropout = 0.4
- Dropout = 0.6

**Compare model performance in terms of:**
- Training loss curves
- Validation macro F1 score
- Test macro F1 score

#### Part B: Robustness Evaluation
Evaluate your model performance by introducing noise in your test data:

**Types of noise to introduce:**
1. **Spelling mistakes** - artificially introduce common spelling errors
2. **Synonym replacement** - replace words with their synonyms

**Analysis:**
- Compare performance on clean vs. noisy test data
- Analyze which dropout value provides better robustness
- Discuss findings on model resilience to noise

---

## Evaluation Metrics

### For Q1 (Image Classification)
- Accuracy
- Macro F1 Score
- Precision and Recall per class
- Confusion Matrix
- Training/Validation curves

### For Q2 (Sentiment Analysis)
- Macro F1 Score (primary metric)
- Training loss curves
- Validation accuracy/F1
- Test accuracy/F1
- Performance degradation analysis with noisy data

---

## Implementation Tasks

### Q1: CNN Implementation
```python
# Suggested Architecture Variations
# Full Architecture: Conv2D -> Pooling -> Conv2D -> Pooling -> FC Layers
# Reduced Architecture: Conv2D -> Pooling -> FC Layers (1 fewer conv layer)
# Further Reduced: FC Layers only (no conv layers)
