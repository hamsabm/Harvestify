import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pickle

# 1. Create your custom dataset (as per your guidelines)
data = {
    'N': [90, 85, 60, 74, 78, 69, 94, 89, 67, 72],
    'P': [42, 58, 55, 35, 42, 37, 53, 54, 45, 51],
    'K': [43, 41, 44, 40, 42, 42, 40, 38, 38, 35],
    'temperature': [20.8, 21.7, 23.0, 26.4, 20.1, 23.0, 24.8, 23.4, 21.3, 25.1],
    'humidity': [82, 80, 82, 80, 81, 82, 81, 80, 83, 80],
    'ph': [6.5, 7.0, 7.8, 6.9, 7.5, 7.0, 6.7, 6.4, 7.2, 6.1],
    'rainfall': [202, 226, 263, 242, 262, 251, 230, 216, 220, 240],
    'label': ['rice', 'rice', 'rice', 'maize', 'rice', 'rice', 'rice', 'rice', 'rice', 'maize']
}

df = pd.DataFrame(data)

# Save this to CSV so you can submit it as part of your project
df.to_csv('my_custom_data.csv', index=False)
print("✅ Custom dataset 'my_custom_data.csv' created.")

# 2. Separate Features and Target
X = df.drop('label', axis=1)
y = df['label']

# 3. Train the model
model = DecisionTreeClassifier()
model.fit(X, y)

# 4. Save the model to the path app.py expects
# Check app.py to see exactly where it loads the model from. 
# Usually, it's in a folder called 'models'
model_path = 'models/RandomForest.pkl' # Or whatever filename app.py uses
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

print(f"✅ New model saved to {model_path}. You can now run app.py!")