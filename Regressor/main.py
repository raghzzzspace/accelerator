import pandas as pd
from models.trainer import get_model
from utils.metrics import regression_metrics
from utils.save_model import save_model
from fe.feature_engineering import feature_engineering
from utils.data_split import train_test_split_data

# Load Data
df = pd.read_csv('data/your_dataset.csv')
X = df.drop('target', axis=1)
y = df['target']

# Split Data
X_train, X_test, y_train, y_test = train_test_split_data(X, y, test_size=0.2, random_state=42)

# Feature Engineering (apply to train and test sets)
X_train_processed = feature_engineering(X_train, method='poly', degree=2)

X_test_processed = feature_engineering(X_test, method='poly', degree=2)


# Model Training
model = get_model('ridge', task='regression')
model.fit(X_train_processed, y_train) # Train on processed training data

# Predict and Evaluate on Test Set
y_pred = model.predict(X_test_processed) # Predict on processed test data

metrics = regression_metrics(y_test, y_pred) # Evaluate using test set ground truth and predictions
print(metrics)

# Save Model
save_model(model, 'ridge_model')