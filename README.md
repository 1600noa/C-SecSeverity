#  Clinical Postpartum Triage: A Cascade NLP Architecture

An advanced Medical Informatics project implementing and evaluating a **Dual-Stage Cascade Architecture** for clinical triage prediction of postpartum patient messages, following the **Manchester Triage System (MTS)**. 

This repository compares localized custom deep learning configurations against large language models (LLMs) in a zero-shot operational setting.

---

## 📌 Project Overview & Clinical Significance
Postpartum patient triage requires high sensitivity to critical signs while managing the high volume of low-urgency clinical messages. Standard flat multi-class classifiers often fail on minority critical classes due to severe data imbalance. 

To bridge this gap, this project structures the triage classification into a **two-step cascade workflow**:
1. **Stage 1 (Binary Filter):** Segregates incoming messages into **Routine (Level 0)** vs. **Urgent (Levels 1-3)** using a Sigmoid/BCE optimization layer.
2. **Stage 2 (Severity Regressor):** Takes filtered urgent cases and places them on a continuous scale optimized with **Mean Squared Error (MSE)** loss, preventing massive penalties for missing life-threatening symptoms.

---

## 🗺️ Clinical Risk Stratification (Manchester Triage Protocol)
The system maps 8 distinct postpartum clinical warning signs into 3 structured Manchester Triage levels based on medical literature:

* 🔴 **Level 3: Immediate / Life-Threatening**
    * `respiratory_warning` (Acute shortness of breath, chest pain)
    * `bleeding_warning` (Postpartum hemorrhage/severe bleeding)
    * `thromboembolism_warning` (Suspected blood clot, unilateral hot/swollen calf)
* 🟠 **Level 2: Moderate / Urgent**
    * `infection_warning` (High fever, worsening surgical redness, sepsis risk)
    * `severe_pain` (Acute unmanaged abdominal or pelvic pain)
* 🟢 **Level 1: Low Urgency / Routine**
    * `wound_problem` (Incision or suture irritation/drainage issues)
    * `urinary_problem` (UTI symptoms, pain post-catheter)
    * `mood_disorder` (Postpartum depression/anxiety red flags)

---

## ⚙️ Model Architectures Compared

### 1. BioBERT Cascade (Pre-trained Transformer)
* **Base:** `dmis-lab/biobert-v1.1` fine-tuned locally.
* **Pipeline:** Dual-stage inference. Stage 1 operates as a binary filter ($num\_labels=1$ with Sigmoid). Stage 2 functions as a regression layer optimized through $MSE$.
* **Regularization:** Controlled via implicit `load_best_model_at_end=True` to prevent overfitting beyond Epoch 3.

### 2. Custom Bi-LSTM Cascade (Deep Learning Baseline)
* **Architecture:** Built from scratch in PyTorch utilizing a bidirectional LSTM layer (`hidden_dim=64`, `embedding_dim=128`).
* **Optimization:** Stage 1 uses `BCEWithLogitsLoss`; Stage 2 uses `MSELoss`.
* **Regularization:** Engineered via customized conditional Checkpointer for **Manual Early Stopping** at 4 Epochs to secure peak generalizability.

### 3. GPT-4o-mini (LLM Zero-Shot Baseline)
* **Methodology:** Direct 4-class multi-class classification utilizing `response_format={"type": "json_object"}`.
* **Inference:** In-context Zero-Shot prompting with explicit embedded operational criteria without local fine-tuning.

---

## 📊 Data Split & Methodology
To guarantee rigorous scientific benchmarking, all models share a deterministic data setup:
* **Training Set:** 60% (Stratified by original triage level)
* **Validation Set:** 15%
* **Test Set (Unseen Evaluation):** 25% (Comprising over 750 unique synthetic/real clinical interactions)

---

## 📂 Repository Structure
```text
├── notebooks/
│   ├── BioBERT_Cascade_Pipeline.ipynb  # Fine-tuning & Cascade code for BERT
│   ├── BiLSTM_PyTorch_Pipeline.ipynb   # Native PyTorch training for Bi-LSTM
│   └── GPT_ZeroShot_Evaluation.ipynb   # OpenAi API benchmarking script
├── plots/
│   ├── bert_cascade_mse_confusion_matrix.png
│   ├── lstm_cascade_mse_confusion_matrix.png
│   ├── gpt_confusion_matrix.png
│   └── bert_cascade_learning_curves.png
├── DATA /
│   ├── data generating.ipynb
│   └── manchester_postpartum_triage_v1.csv  # Anonymized / Structured Dataset
└── README.md                         
