# Hybrid Sentiment Analysis: Classical ML vs. LLMs
A comprehensive comparative study on sentiment classification using **Classical Machine Learning (SVM + BGE Embeddings)** and **Large Language Models (Llama 3.1)**. 
## Overview
This project evaluates different methodologies for binary sentiment classification on the IMDb movie reviews dataset. It explores the synergy between state-of-the-art embedding models and traditional classifiers, contrasted against the reasoning capabilities of modern LLMs via Zero-shot and Few-shot learning.
## Key Features
- **Comparative Analysis**: Head-to-head comparison between Supervised Learning (SVM) and In-Context Learning (LLMs).
- **Advanced Embeddings**: Utilizes **BAAI BGE-small-en-v1.5** for high-quality semantic representation.
- **LLM Integration**: Leverages **Llama 3.1-8B-Instant** (via Groq) for high-speed inference.
- **Prompt Engineering**: Implements **Few-shot learning** with **MMR (Max Marginal Relevance)** example selection to optimize context relevance.
- **Performance Optimization**: Strategically optimized context windows through smart truncation and sample size selection ($k=3$).
## Technology Stack
- **Frameworks**: LangChain, Scikit-learn
- **LLM Engine**: Groq API (Llama 3.1)
- **Vector DB/Search**: FAISS (for MMR selector)
- **Embeddings**: HuggingFace (BGE-small)
- **Data Engineering**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
## Methodology
### 1. Classical Machine Learning (SVM + BGE)
- **Feature Extraction**: 384-dimensional semantic embeddings generated via `bge-small-en-v1.5`.
- **Optimization**: Hyperparameter tuning using `GridSearchCV` (Linear Kernel, optimized $C$).
- **Scale**: Evaluated on a 1,000-sample independent test set.
### 2. Large Language Model (Llama 3.1)
- **Zero-shot**: Strict persona-based instructions ("Expert Film Critic").
- **Few-shot (MMR)**: Dynamic retrieval of the top 3 most semantically diverse and relevant examples using MMR selector.
- **Context Management**: Smart truncation to 1000 characters to balance context depth and model focus.
## Results
The study achieved a high level of accuracy across all methods, with LLMs showing exceptional zero-shot capabilities and SVM showing robust scalability.
| Metric | LLM (Zero/Few-shot) | SVM + BGE-small |
| :--- | :---: | :---: |
| **Accuracy** | **94.0%** | **93.8%** |
| **Weighted F1** | **94.04%** | **93.80%** |
| **Balanced Acc** | **0.9483** | **0.9381** |
### Key Insights:
- **Optimization Impact**: Reducing Few-shot examples from 5 to 3 and implementing text truncation improved LLM performance to match Zero-shot parity (94%).
- **Semantic Precision**: Both LLM variants achieved **zero errors** in detecting negative sentiment within the evaluation sample.
- **Cost-Efficiency**: SVM with BGE embeddings provides near-identical accuracy to LLMs while being significantly more efficient for large-scale production tasks.
## Conclusion
This project demonstrates that while LLMs provide the highest accuracy with zero manual training, **traditional classifiers like SVM, when paired with modern embeddings (BGE), remain highly competitive** and production-ready for fixed NLP tasks.

