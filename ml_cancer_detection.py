import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt

# Seed for reproducibility
np.random.seed(42)

# Number of samples
num_samples = 100000

# Generate synthetic data with stronger correlations
age = np.random.randint(20, 80, size=num_samples)
tumor_size = np.random.uniform(0.1, 10.0, size=num_samples)
genetic_marker_1 = np.random.choice([0, 1], size=num_samples, p=[0.7, 0.3])
genetic_marker_2 = np.random.choice([0, 1], size=num_samples, p=[0.6, 0.4])
genetic_marker_3 = np.random.choice([0, 1], size=num_samples, p=[0.5, 0.5])
genetic_marker_4 = np.random.choice([0, 1], size=num_samples, p=[0.4, 0.6])
genetic_marker_5 = np.random.choice([0, 1], size=num_samples, p=[0.3, 0.7])
family_history = np.random.choice([0, 1], size=num_samples, p=[0.7, 0.3])

# Target variable with stronger correlation
cancer = (0.3*genetic_marker_1 + 0.4*genetic_marker_2 + 0.5*genetic_marker_3 +
          0.6*genetic_marker_4 + 0.7*genetic_marker_5 + 0.8*family_history +
          np.random.normal(0, 0.1, size=num_samples)) > 1.5
cancer = cancer.astype(int)

# Create DataFrame
df = pd.DataFrame({
    'age': age,
    'tumor_size': tumor_size,
    'genetic_marker_1': genetic_marker_1,
    'genetic_marker_2': genetic_marker_2,
    'genetic_marker_3': genetic_marker_3,
    'genetic_marker_4': genetic_marker_4,
    'genetic_marker_5': genetic_marker_5,
    'family_history': family_history,
    'cancer': cancer
})

# Split the data into features and target variable
X = df.drop('cancer', axis=1)
y = df['cancer']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the feature variables
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the model
model = RandomForestClassifier(random_state=42)

# Train the model
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Calculate ROC AUC
fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test_scaled)[:, 1])
roc_auc = auc(fpr, tpr)

# Print evaluation metrics
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

# Plot ROC Curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
