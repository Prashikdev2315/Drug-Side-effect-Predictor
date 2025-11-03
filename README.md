# Drug Side Effect Prediction Using Machine Learning  

## 📌 Overview  
This project predicts the **side effects of drugs** using either their **name** or **SMILES (Simplified Molecular Input Line Entry System)** structure.  
If a drug already exists in the dataset, users can simply provide its **name** to retrieve known side effects.  
For **new or unknown drugs**, users can input the **SMILES string**, and the trained **Multi-Layer Perceptron (MLP)** model predicts the most probable side effects based on molecular structure.  

The system includes a **simple frontend interface** where users can enter the drug details, select how many side effects they wish to view, and instantly see the **predictions displayed clearly on screen** with **probability bars** and **severity indicators** (e.g., *Medium*, *High*).  

---

## 🚀 Features  
- Predict side effects from **drug name** or **SMILES structure**  
- Interactive **frontend interface** for easy user input and output visualization  
- Displays predictions ranked by **probability**  
- Shows **probability values**, **thresholds**, and **severity labels**  
- MLP-based deep learning model trained on reliable biomedical datasets  
- Supports both **known** and **novel** drugs  

---

## 📂 Project Structure  
```
.
├── app.py                     # Entry point to run the project
├── model/                     # Trained MLP model files
├── data/                      # Preprocessed datasets (STITCH, SIDER, DrugBank)
├── notebooks/                 # Model training and analysis notebooks
├── utils/                     # Helper scripts (data cleaning, feature extraction)
├── templates/                 # HTML templates for frontend interface
├── static/                    # CSS/JS files for UI styling
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

The MLP model was trained on molecular feature vectors derived from SMILES strings using **RDKit** and statistical feature extraction techniques.

---

## 🖥️ Usage Flow  
1. Open the web interface.  
2. Enter the **drug name** or **SMILES structure** in the input field.  
3. Select the **number of side effects** to view (e.g., Top 3, Top 5, Top 10).  
4. Click **Predict**.  
5. The system displays results in an organized list format, each containing:  
   - Side effect name  
   - Probability percentage  
   - Defined threshold value  
   - Severity level (e.g., *Low*, *Medium*, *High*)  
   - Horizontal probability bar showing confidence visually  

Example (textual format):  
```
1. Rash — Probability: 66.07% | Threshold: 15% | Severity: Medium  
2. Dermatitis — Probability: 63.57% | Threshold: 15% | Severity: Medium  
3. Nausea — Probability: 63.20% | Threshold: 15% | Severity: Medium  
```

---

## 🌍 Future Scope  
- Integration with **human gene data** for personalized side effect analysis.  
- Use of **Graph Neural Networks (GNNs)** for improved molecular representation.  
- Deployment to **web/cloud platforms** for wider accessibility.  
- Support for **drug-drug interaction** prediction.

---

## ⚠️ Disclaimer  
This project is developed for **academic and research purposes only**.  
It is **not a medical diagnostic tool** and should not replace professional medical advice.  
