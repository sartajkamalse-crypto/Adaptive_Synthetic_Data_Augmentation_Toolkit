# Adaptive_Synthetic_Data_Augmentation_Toolkit

An iterative, closed-loop Machine Learning framework developed during my internship at **ITSOLERA Pvt. Ltd.** to dynamically handle extreme class imbalance using generative data techniques.

## Project Overview
Real-world datasets in critical fields like finance and healthcare often suffer from severe class imbalance, causing models to fail on minority classes. This project introduces an Adaptive Synthetic Data Augmentation Toolkit that implements a closed-loop feedback mechanism. Instead of static augmentation, it actively identifies model vulnerabilities (specifically false negatives) and generates targeted synthetic data to iteratively improve performance.

## Tools and Technologies
* **Programming Language:** Python
* **Generative Model Library:** SDV (Synthetic Data Vault)
* **Machine Learning Library:** Scikit-learn
* **Data Manipulation & Visualization:** Pandas, NumPy, Matplotlib, Seaborn

---

## 📊 Dataset Description
The toolkit is validated using a real-world **Credit Card Fraud Detection dataset** consisting of **227,845 training samples** and 30 features:
* **Class 0 (Normal Transactions):** 227,451 samples
* **Class 1 (Fraudulent Transactions):** 394 samples (Highly imbalanced)

---

## ⚙️ Methodology & Closed-Loop Pipeline

### 1. Baseline Model Execution
A Random Forest Classifier was trained using stratified splits to establish initial benchmarks, revealing initial difficulties in identifying rare fraud cases.

### 2. Error Analysis
The system scans predictions to isolate false negatives (critical fraudulent cases incorrectly flagged as normal) to prioritize them for generation.

### 3. Targeted Synthetic Generation
The pipeline leverages a **Gaussian Copula Synthesizer** from the SDV library, restricting synthetic generation exclusively to minority class error samples so the model explicitly learns from its previous mistakes.

### 4. Iterative Retraining Loop
1. Train the classifier.
2. Extract false negative instances.
3. Generate focused synthetic data samples.
4. Augment the training dataset.
5. Retrain the model and evaluate performance.
6. Repeat the cycle until performance plateaus.

---

## 📈 Performance & Results

The adaptive framework successfully minimized false negatives and significantly raised the model's sensitivity to the minority class across successive iterations.

* **Baseline Recall (Class 1):** 0.7449
* **Final Iteration Recall (Class 1):** **0.8571**
* **Total Metric Boost:** ⭐ **+11.22% improvement** in minority class detection

### Confusion Matrix Progress:
| Iteration Stage | True Negatives | False Positives | False Negatives | True Positives |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline Model** | 56,861 | 3 | 25 | 73 |
| **Iteration 1** | 56,856 | 8 | 15 | 83 |
| **Iteration 2** | 56,853 | 11 | 14 | 84 |
| **Iteration 3** | 56,853 | 11 | **14** | **84** |

---

## ⚠️ Challenges and Solutions
1. **Stagnant Initial Progress:** Resolved by restricting the synthetic data generation engine to focus exclusively on minority class errors rather than broad generation.
2. **Majority Class Bias:** Counteracted by assigning balanced penalty weights via `class_weight='balanced'`.
3. **Overfitting Risks:** Managed using strictly controlled augmentation volumes alongside early stopping thresholds.
