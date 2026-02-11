# Insurance-and-Heart-Disease-ML-Project-part-1
End-to-end Machine Learning preprocessing and feature engineering project
🧠 Insurance & Heart Disease Data Analysis Project
📌 Project Overview

This project focuses on data preprocessing, exploratory data analysis (EDA), feature extraction, and feature selection performed on two real-world datasets:

🏥 Heart Disease Dataset

💰 Insurance Dataset

The goal of this project is to understand the data thoroughly and prepare it for machine learning by performing systematic data cleaning and transformation techniques.
This project mainly emphasizes data preparation and analysis, not model building.

📂 Datasets Used
1️⃣ Heart Disease Dataset
Contains medical attributes used to analyze heart disease risk.
Some features include:
-->Age
-->Sex
-->Chest Pain Type
-->Resting Blood Pressure
-->Cholesterol
-->Fasting Blood Sugar
-->Maximum Heart Rate
-->Target (Presence of heart disease)

2️⃣ Insurance Dataset
Contains personal and health-related attributes used to analyze insurance charges.
Features include:
-->Age
-->Sex
-->BMI
-->Number of Children
-->Smoker
-->Region
-->Charges

🔍 Project Workflow
The project follows a structured data analysis pipeline:
1️⃣ Data Cleaning
Data cleaning is the process of detecting and correcting inaccurate or missing data.
Steps Performed:

..>Checked for missing values using:

df.isnull().sum()

Removed or handled null values where necessary.
Checked for duplicate records:

df.duplicated().sum()


Removed duplicates if present.
Corrected inconsistent categorical values.

2️⃣ Data Preprocessing

Data preprocessing transforms raw data into a format suitable for analysis.
✔ Encoding Categorical Variables
Used Label Encoding or One-Hot Encoding:

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])
or
df = pd.get_dummies(df, drop_first=True)

✔ Feature Scaling
Standardization or normalization was applied where necessary:

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

3️⃣ Exploratory Data Analysis (EDA)

EDA helps understand patterns, trends, and relationships in the data.

✔ Statistical Summary
df.describe()


Provides:
Mean
Standard Deviation
Min & Max
Quartiles

✔ Correlation Analysis
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()


Used to identify relationships between variables.

✔ Data Visualization
Some visualizations used:
Histograms
Count plots
Box plots
Scatter plots

Example:

sns.countplot(x='sex', data=df)

4️⃣ Feature Extraction

Feature extraction is the process of creating new meaningful features from existing data.

Examples:

Creating BMI categories
Transforming categorical columns
Combining related variables

Purpose:
Improve data representation
Enhance analytical understanding

5️⃣ Feature Selection

Feature selection involves selecting the most important features for analysis.

✔ Correlation-based selection

Removed highly correlated features to avoid multicollinearity.

✔ Removing irrelevant features

Dropped unnecessary columns:

df.drop(columns=['unnecessary_column'], inplace=True)


Goal:

Reduce dimensionality
Improve data quality
Prepare data for future modeling

📊 Technologies Used

Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
Jupyter Notebook

🎯 Key Learnings

Through this project, I learned:

Real-world data cleaning techniques

Handling categorical and numerical data

Performing structured EDA

Understanding feature importance

Preparing datasets for machine learning pipelines

🚀 Future Scope

Implement classification models for heart disease prediction

Implement regression models for insurance charge prediction

Hyperparameter tuning

Model evaluation and deployment
