# 🌸 Iris Dataset — Exploratory Data Analysis

> **CodeAlpha Data Science Internship — Task 1**

---

## 📌 Overview

This project performs a complete **Exploratory Data Analysis (EDA)** and **Data Visualization** on the classic [Iris flower dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set). The dataset contains 150 samples across 3 species of Iris flowers, with 4 features measured for each sample.

---

## 📂 Project Structure

```
codealpha_iris/
│
├── iris_analysis.py      # Main EDA and visualization script
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
└── plots/                # Generated visualizations (auto-created on run)
    ├── 01_feature_distributions.png
    ├── 02_boxplots.png
    ├── 03_violinplots.png
    ├── 04_pairplot.png
    ├── 05_correlation_heatmap.png
    ├── 06_petal_scatter.png
    ├── 07_sepal_scatter.png
    └── 08_class_distribution.png
```

---

## 📊 Dataset Info

| Property       | Details                              |
|----------------|--------------------------------------|
| Samples        | 150                                  |
| Features       | 4 (sepal length, sepal width, petal length, petal width) |
| Classes        | 3 (Setosa, Versicolor, Virginica)    |
| Missing Values | None                                 |
| Source         | `sklearn.datasets.load_iris()`       |

---

## 🛠️ Tech Stack

- **Python 3.x**
- **Pandas** — data manipulation
- **NumPy** — numerical operations
- **Matplotlib** — base plotting
- **Seaborn** — statistical visualizations
- **Scikit-learn** — dataset loading

---

## ▶️ How to Run

**1. Clone the repository**
```bash
git clone https://github.com/chandub-debug/codealpha_iris.git
cd codealpha_iris
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the analysis**
```bash
python iris_analysis.py
```

All 8 plots will be saved automatically in the `plots/` folder.

---

## 📈 Visualizations Generated

| # | Plot | Description |
|---|------|-------------|
| 1 | Feature Distributions | Histogram + KDE for each feature by species |
| 2 | Box Plots | Spread and outliers per feature per species |
| 3 | Violin Plots | Distribution shape + quartiles per species |
| 4 | Pair Plot | All feature combinations colored by species |
| 5 | Correlation Heatmap | Feature correlation matrix |
| 6 | Petal Scatter | Petal length vs petal width |
| 7 | Sepal Scatter | Sepal length vs sepal width |
| 8 | Class Distribution | Count of samples per species |

---

## 🔍 Key Insights

- **Petal features** (length & width) are the most discriminative — Setosa is clearly separable from the other two species.
- **Sepal width** shows the least separation between classes.
- Strong positive correlation (**≈ 0.96**) between petal length and petal width.
- Sepal length and petal length are moderately correlated (**≈ 0.87**).
- Sepal width is slightly **negatively** correlated with the other three features.
- Versicolor and Virginica have some overlap, especially in sepal measurements.

---

## 👤 Author

**Chandu B**  
CodeAlpha Data Science Internship  

---
