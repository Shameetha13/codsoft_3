import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
ds = pd.read_csv("iris.csv")

# Check for missing values
print("Missing values in the dataset:\n", ds.isnull().sum())

# Visualize class distribution
plt.figure()
ds['species'].value_counts().plot(kind='pie', autopct='%1.2f%%')
plt.title('Distribution of Iris Species')
plt.show()

# Visualize feature distributions
plt.figure(figsize=(12, 6))
for i, feature in enumerate(ds.columns[:-1]):
    plt.subplot(2, 2, i + 1)
    sns.histplot(data=ds, x=feature, hue='species', kde=True)
plt.tight_layout()
plt.show()

# Encode the categorical target variable 'species'
label_encoder = LabelEncoder()
ds['species'] = label_encoder.fit_transform(ds['species'])

# Split the dataset into train and test sets
train, test = train_test_split(ds, test_size=0.25, random_state=42)

# Define features and target variable
features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
target = 'species'

# Train and evaluate models
models = {
    'Logistic Regression': LogisticRegression(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Support Vector Machine': SVC(),
    'Naive Bayes': GaussianNB(),
    'Decision Tree': DecisionTreeClassifier()
}

for name, model in models.items():
    model.fit(train[features], train[target])
    y_pred = model.predict(test[features])
    acc = accuracy_score(test[target], y_pred)
    print(f"Model: {name}, Accuracy: {acc:.4f}")
    print(classification_report(test[target], y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(test[target], y_pred))
    print("="*50)

# Visualize feature importance for Decision Tree
if 'Decision Tree' in models:
    dt_model = models['Decision Tree']
    feature_importance = pd.Series(dt_model.feature_importances_, index=features)
    plt.figure(figsize=(8, 5))
    feature_importance.sort_values(ascending=False).plot(kind='bar')
    plt.title('Feature Importance (Decision Tree)')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.show()
