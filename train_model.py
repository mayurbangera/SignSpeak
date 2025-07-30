import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Path to your data folder
data_path = 'data'

all_data = []
all_labels = []

# Load all CSV files
for file in os.listdir(data_path):
    if file.endswith('.csv'):
        label = file.replace('.csv', '')
        df = pd.read_csv(os.path.join(data_path, file), header=None)
        for row in df.values:
            all_data.append(row[:-1])   # All landmarks (X, Y, Z)
            all_labels.append(label)    # Label like Hello, Yes

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(all_data, all_labels, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Check accuracy (optional)
accuracy = model.score(X_test, y_test)
print(f"✅ Model trained with accuracy: {accuracy * 100:.2f}%")

# Save the model to a file
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model saved as model.pkl")
