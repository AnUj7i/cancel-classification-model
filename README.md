# 🧬 Cancer Cell Classification using Scikit-learn

This project classifies breast cancer cells as **Malignant (0)** or **Benign (1)** using the **Breast Cancer Wisconsin dataset** from Scikit-learn.  
It includes an **interactive EDA dashboard built with Streamlit**, and a trained **Random Forest classifier** to predict cancer type based on cell features.

---

## 🚀 Live Demo

🎯 **Streamlit App on Hugging Face Spaces**  
[👉 Click here to view the app]()  
*(Replace with your actual Hugging Face link)*

## Viswals 

![Screenshot 2025-07-08 164255](https://github.com/user-attachments/assets/e4c805c7-6c9f-4ccc-9d35-804c29427285)

# plot
![Screenshot 2025-07-08 163951](https://github.com/user-attachments/assets/ea3482dd-f501-4412-beee-1818ec883a58)


# correlation
![Screenshot 2025-07-08 164012](https://github.com/user-attachments/assets/823b3194-af29-408b-8a0a-b588197442ec)


## 📌 Project Highlights

- ✅ Clean EDA visualizations with **Plotly**, **Seaborn**, and **Matplotlib**
- ✅ Real-time user input prediction using **Random Forest**
- ✅ Deployable web dashboard using **Streamlit**
- ✅ Packaged with `requirements.txt` for reproducibility
  

---

## 📊 Dashboard Features

| Feature | Description |
|--------|-------------|
| 📈 Histogram Viewer | Explore distributions of features (e.g., `mean radius`) |
| 🧬 Class Distribution | Visualize Benign vs Malignant cells |
| 🔍 Correlation Heatmap | Analyze feature correlations |
| 🧠 Classifier | Predict cancer type from user-provided values |

---

## 📁 File Structure

```bash
cancer-classification/
├── app.py                # Streamlit dashboard + prediction
├── model.py  # Train and evaluate the ML model
├── eda.py    # data analysis dashboard
├── requirements.txt      # Python dependencies
└── README.md             # You're reading this
