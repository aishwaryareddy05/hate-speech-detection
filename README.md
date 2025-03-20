# 🔎 Hate Speech Detection System

## 📌 Overview
This project is an **AI-powered hate speech detection system** that classifies text into three categories:
- 🔴 **Hate Speech**
- ⚠️ **Offensive Language**
- ✅ **No Hate and Offensive**

The model is trained using **Logistic Regression** with **TF-IDF vectorization** for feature extraction. It incorporates **SMOTE** to handle class imbalances and **SVD** for dimensionality reduction. A **Streamlit app** is included for real-time text classification.

## 🌟 Features
- 🎭 **Real-time hate speech detection** using an interactive web interface
- 🎯 **Optimized predictions** with an adjusted threshold (0.6) to reduce false positives
- 🔍 **Preprocessing pipeline**: Tokenization, stopword removal, lemmatization, and text normalization
- 📊 **Performance Metrics**: Accuracy score, classification report, and confusion matrix
- 💾 **Model Training and Persistence**: Model, vectorizer, and transformer saved using `joblib`

## 📂 Dataset
The dataset used is from **Twitter**, containing labeled tweets categorized as:
- `0` → 🔴 Hate Speech
- `1` → ⚠️ Offensive Language
- `2` → ✅ No Hate and Offensive

## 🛠 Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/hate-speech-detection.git
   cd hate-speech-detection
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Download NLTK data**:
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```
4. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

## 🎓 Model Training
To train the model, run:
```bash
python train.py
```
This will preprocess the data, train the model, and save the trained artifacts (`log_reg_model.pkl`, `tfidf_vectorizer.pkl`, `svd_transform.pkl`).

## 🌐 Streamlit Web App
The **Streamlit app** allows users to input text and classify it in real-time. Simply run:
```bash
streamlit run app.py
```

## 📊 Results
- **Accuracy**: 🎯 84%
- **Precision, Recall, F1-Score**: See `classification_report`
- **Confusion Matrix**: Displayed in the terminal

### 📌 **Sample Predictions**
| 📝 Text                          | 🎯 Prediction         |
|--------------------------------|-------------------|
| I hate this person so much!    | 🔴 Hate Speech       |
| You are so dumb and useless!   | 🔴 Hate Speech       |
| Have a great day everyone!     | ✅ No Hate and Offensive |
| That was an awesome match!     | ✅ No Hate and Offensive |
| I want to kill him             | 🔴 Hate Speech       |
| I don't like her               | ⚠️ Offensive Language |

## 🤝 Contributing
Feel free to fork this repository and submit **pull requests** for improvements.

## 📜 License
This project is licensed under the **MIT License**.

## 👤 Author
[Aishwarya Bojja](https://github.com/aishwaryareddy05)


