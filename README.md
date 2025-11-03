# Drug Side Effect Prediction Using Machine Learning  

## 📌 Overview  
This project focuses on predicting the **side effects of drugs** using either their **name** or **SMILES (Simplified Molecular Input Line Entry System)** structure.  
If a drug already exists in the dataset, users can provide its **name** to retrieve the known side effects.  
For **new or unknown drugs**, users can input the **SMILES string**, and the trained **Multi-Layer Perceptron (MLP)** model will predict the most probable side effects based on molecular structure.  

Users can also choose **how many side effects** they want to view, and results are displayed in **decreasing order of probability**.  

---

## 🚀 Features  
- Predict side effects from either **drug name** or **SMILES structure**  
- Uses **MLP-based deep learning model** trained on verified biomedical datasets  
- Displays side effects ranked by **probability of occurrence**  
- Simple and interactive **app interface**  
- Handles both **known** and **novel** drug inputs  

---

## 📂 Project Structure  
```
.
├── app.py                     # Entry point to run the project
├── model/                     # Trained MLP model files
├── data/                      # Preprocessed datasets (STITCH, SIDER, DrugBank)
├── notebooks/                 # Model training and analysis notebooks
├── utils/                     # Helper scripts (data cleaning, feature extraction)
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## ⚙️ Installation  

### 1️⃣ Clone the repository  
```bash
git clone https://github.com/your-username/Drug-Side-effect-Predictor.git
cd Drug-Side-effect-Predictor
```

### 2️⃣ Create and activate a virtual environment  
```bash
python -m venv venv
source venv/bin/activate    # Linux / macOS
venv\Scripts\activate       # Windows
```

### 3️⃣ Install dependencies  
```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application  
Once dependencies are installed, simply run:  
```bash
python app.py
```

Then open the application in your browser:  
👉 `http://127.0.0.1:5000/`

---

## 🧠 Model and Dataset Details  

- **Model Type:** Multi-Layer Perceptron (MLP)  
- **Libraries Used:**  
  - PyTorch  
  - NumPy  
  - Pandas  
  - Scikit-learn  
  - RDKit  
  - tqdm  
  - Joblib  
  - Matplotlib  
  - Seaborn  
- **Datasets Used:**  
  - **STITCH** — Chemical-protein interaction data  
  - **SIDER** — Documented side effects of approved drugs  
  - **DrugBank** — Drug and chemical compound database  

The MLP model was trained on molecular feature vectors derived from SMILES strings using **RDKit** and statistical feature engineering methods.

---

## 🖥️ Usage Flow  
1. Enter the **drug name** or **SMILES structure**.  
2. Select how many side effects you want to view (e.g., Top 5 / Top 10).  
3. Click **Predict**.  
4. The system will display:  
   - List of predicted side effects  
   - Corresponding probabilities (in descending order)  

---

## 📊 Expected Output Example  
```
Input Drug: Tamoxifen
Top 5 Predicted Side Effects:
1. Drowsiness (0.89)
2. Headache (0.86)
3. Vision Blurred (0.80)
4. Tachycardia (0.73)
5. Dizziness (0.70)
```

---

## 🌍 Future Scope  
- Integration with **human gene data** to study personalized side effect predictions.  
- Enhanced molecular feature extraction using **Graph Neural Networks (GNNs)**.  
- Deployment on **web and cloud platforms** for real-time access.  
- Expansion to include **drug-drug interaction** effects.

---

## ⚠️ Disclaimer  
This project is developed for **academic and research purposes only**.  
It is **not a medical diagnostic system** and should not replace professional medical advice.  
