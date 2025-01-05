# CS445 Project: Natural Language Processing

This repository contains the project for the CS445 - Natural Language Processing course. 

## Overview
Our project explores intent recognition and slot filling, key tasks in natural language understanding. We conducted two experiments, as outlined below:

### Experiments
1. **Baseline Model**: Our baseline with Naive Bayes and TF-IDF achieved an accuracy of **93%**.
2. **Logistic Regression with TF-IDF**: Achieved an accuracy of **93%**.
3. **Logistic Regression with Bert**: Achieved an accuracy of **94%**.
4. **TF-IDF unigram/bigram/trigram Comparison with SVM**: Unigram, Bigram, Trigram Achieved accuracies of **92.95%, 93.73%, 92.50%** respectively.
5. **Impact of Oversampling with TF-IDF (Unigram) and SVM**: Achieved an accuracy of **93.06%** which is worse than without oversampling.
6. **Bi-model with Slot-filling**: Authors of this method received an accuracy of **98%**. 
7. **Slot-Gated Modeling for Joint Slot Filling and Intent Prediction**: Authors of this method received an accuracy of **93%**.
8. **Improved Slot-Gated Modelling for Intent Prediction**: We have an accuracy of **96%** compared to the third model.
   
## References
- [Bi-Model for Intent and Slot](https://github.com/ray075hl/Bi-Model-Intent-And-Slot/tree/master)
- [ATIS Dataset](https://github.com/howl-anderson/ATIS_dataset/tree/master/data)
- [Slot-gated SLU](https://github.com/MiuLab/SlotGated-SLU)
---

We aim to further improve these results and compare with state-of-the-art approaches.
