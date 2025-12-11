
# Flight Delay Data Cleaning & Feature Engineering Project

## 📌 Overview
This project focuses exclusively on **data cleaning**, **feature engineering**, and **preprocessing** of the U.S. Department of Transportation flight delay dataset (5.8M rows).  
It is designed to demonstrate strong skills in:

- Real-world data handling  
- Cleaning messy, high-volume data  
- Correctly dealing with outliers & missing values  
- Feature engineering for ML readiness  
- Writing production-grade analysis notebooks  
- Structuring a professional AI/data project

This project is ideal for showcasing capability as a **Senior Software Engineer**, **Machine Learning Engineer**, or **AI Developer**.

---

## 📂 Project Structure

```
flight-delay-cleaner/
│
├── data/
│   ├── raw/
│   │   └── flights_raw.csv
│   ├── cleaned/
│   │   └── flights_cleaned.csv
│
├── notebooks/
│   └── 01_data_cleaning.ipynb
│
├── src/
│   ├── __init__.py
│   ├── load_data.py
│   ├── clean_data.py
│   ├── feature_engineering.py
│   └── utils.py
│
├── models/
│   └── baseline_model.pkl   (optional)
│
├── README.md
└── requirements.txt

```

---

## 🧼 Step 1 — Data Cleaning

### 1. Duplicate Removal
Duplicates distort averages, inflate counts, and cause models to overweight repeated rows.

```python
df = df.drop_duplicates()
```

### 2. Fixing Data Types
Convert numeric columns stored as strings and date columns stored as integers:

```python
df['FL_DATE'] = pd.to_datetime(df['FL_DATE'], errors='coerce')
```

### 3. Missing Values
Different strategies applied depending on the variable type:

- **Categorical** → fill with `"Unknown"`
- **Numeric delay components** → fill with median
- **Critical fields** (flight number, airport codes) → drop row

### 4. Handling Impossible Values
Some flights show *negative* delays or durations.  
These violate physical reality and are removed.

```python
df = df[df['DEPARTURE_DELAY'] >= 0].copy()
df = df[df['ARRIVAL_DELAY'] >= 0].copy()
```

### 5. Outlier Removal (IQR)
Delays over 1,000 minutes are extremely rare and harmful to model training.  
Handled using the Interquartile Range rule.

---

## 🛠 Step 2 — Feature Engineering

### Extracted Features
- **day_of_week**
- **month**
- **is_weekend**
- **One-hot encoded airline**
- **One-hot encoded origin airport**
- **One-hot encoded destination airport**

### Why These Features?
They correlate directly with known delay patterns:

- Weekends vs weekdays  
- Seasonal effects  
- Airline performance differences  
- Airport congestion differences  

### One-Hot Encoding
```python
df = pd.get_dummies(
    df,
    columns=['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT'],
    drop_first=True
)
```

---

## 📈 Outputs
- Cleaned dataset stored as **Parquet** (best for large datasets)
- Summary statistics
- Before/after comparisons
- Histograms of cleaned variables

---

## ⚡ Why This Project Matters
This project demonstrates:

- Senior-level understanding of real-world messy datasets  
- Reusable, modular, scalable data processing code  
- Best practices in dataset sanitation  
- Awareness of leakage, overfitting, and ML-readiness  
- Strong data engineering discipline  

This is the type of preprocessing work done at airlines, fintech companies, logistics companies, and ML-heavy organizations.

---

## 🚀 How to Run the Project

1. Clone the repo:
```bash
git clone https://github.com/<yourusername>/flight-delay-cleaner.git
cd flight-delay-cleaner
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Open the notebooks:
```bash
jupyter notebook
```

---

## 📝 Future Improvements
- Add weather data from NOAA  
- Add holiday feature engineering  
- Use airline operational metrics  
- Try embedding layers for high-cardinality airports  
- Build predictive models (XGBoost, CatBoost, LightGBM)

---

## 📄 License
MIT License
git remote add origin https://github.com/anjayluh/flight-delay-cleaner.git
