# Autoregressive Word-Level Text Generation: SimpleRNN vs BiRNN vs LSTM vs GRU

**Course:** Foundations of Generative AI (ICT4403), Assignment 2[cite: 9, 10]  
**Student:** Aryan Saxena  
**Registration Number:** 220953552  

---

## 1. Problem Statement and Objective

The objective of this assignment is to develop and evaluate an autoregressive text-generation system that predicts the next word given a sequence of prior words[cite: 8, 9, 10]. To understand the impact of architectural choices on sequential learning, four distinct recurrent architectures are trained under identical conditions:
1. **SimpleRNN**
2. **Bidirectional RNN (BiRNN)**
3. **Long Short-Term Memory (LSTM)**[cite: 8, 10]
4. **Gated Recurrent Unit (GRU)**[cite: 8, 10]

The core goals are to preprocess a text corpus into word-level sliding-window sequences, quantitatively compare models via loss, perplexity, and training efficiency, and qualitatively analyze text generation under varying sampling temperatures[cite: 8, 9, 10].

---

## 2. Dataset

| Property | Detail |
| :--- | :--- |
| **Dataset Name** | *Alice's Adventures in Wonderland* by Lewis Carroll[cite: 9, 10] |
| **Source** | Project Gutenberg, eBook #11[cite: 9, 10] |
| **Dataset URL** | [Project Gutenberg eBook #11](https://www.gutenberg.org/ebooks/11)[cite: 9, 10] |
| **Direct Text Link** | [Cache Text Link](https://www.gutenberg.org/cache/epub/11/pg11.txt)[cite: 10] |
| **Preprocessing Steps** | Removed Gutenberg header/footer metadata, converted text to lowercase, normalized curly quotes to straight apostrophes, collapsed whitespaces, and tokenized words alongside punctuation marks (`.`, `,`, `!`, `?`, `;`, `:`). Rare words occurring fewer than 2 times were mapped to `<unk>`. |
| **Sequence Windows** | Sliding window with a context length of 10 words as input and the 11th word as the target (stride = 1). |
| **Data Split** | Chronological split: First 90% for training, last 10% for validation (split prior to windowing to avoid data leakage). |

---

## 3. Model Architectures & Experimental Setup

All four models share an identical structural skeleton to ensure a fair, comparable evaluation:
`Embedding (128) -> Recurrent Layer (128 units) -> Dropout (0.3) -> Dense (256, ReLU) -> Dropout (0.3) -> Dense (Vocab Size, Softmax)`

* **Optimizer:** Adam (Learning Rate = $1 \times 10^{-3}$, gradient clipping norm = 1.0)
* **Loss Function:** Sparse Categorical Cross-Entropy
* **Batch Size / Epochs:** Batch size of 128, trained up to 30 epochs with Early Stopping on validation loss (patience = 3, restoring best weights)[cite: 10]

---

## 4. Execution Steps

### **Google Colab (Recommended)**
1. Open the `FGAI_Assignment2_Text_Generation.ipynb` notebook in Google Colab[cite: 10].
2. Navigate to **Runtime > Change runtime type** and select **GPU (T4)**[cite: 10, 11].
3. Click **Runtime > Run all**[cite: 10, 11]. The script automatically fetches the dataset, trains all models, evaluates metrics, runs temperature sampling, and packs outputs into an archive[cite: 10, 11].

### **Local Execution**
```bash
git clone [https://github.com/aryanpsaxena/Autoregressive-Text-Generation-RNN-LSTM-GRU.git](https://github.com/aryanpsaxena/Autoregressive-Text-Generation-RNN-LSTM-GRU.git)
cd Autoregressive-Text-Generation-RNN-LSTM-GRU
pip install tensorflow numpy pandas matplotlib ipython
python fgai_assignment2.py
