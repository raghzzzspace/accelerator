import pandas as pd
from sklearn.model_selection import GridSearchCV
from models.trainer import get_model
from utils.metrics import regression_metrics
from utils.save_model import save_model
from fe.feature_engineering import feature_engineering
from utils.data_split import train_test_split_data

df = pd.read_csv('dataset.csv')
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split_data(X, y, test_size=0.2, random_state=42)

fe_transformer = feature_engineering(X_train, method='poly', degree=2, return_transformer=True)
X_train_processed = fe_transformer.transform(X_train)
X_test_processed = fe_transformer.transform(X_test)

model = get_model('ridge', task='regression')

param_grid = {
    'alpha': [0.01, 0.1, 1, 10, 100],
    'solver': ['auto', 'svd', 'cholesky', 'lsqr']
}

def tune_model(model, param_grid, X_train, y_train, scoring='r2', cv=5):
    grid = GridSearchCV(model, param_grid, scoring=scoring, cv=cv, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_

tuned_model, best_params = tune_model(model, param_grid, X_train_processed, y_train)
print("🔧 Best Hyperparameters:", best_params)

y_pred = tuned_model.predict(X_test_processed)
metrics = regression_metrics(y_test, y_pred)
print("Evaluation Metrics:", metrics)

import joblib
joblib.dump((tuned_model, fe_transformer), 'ridge_model_tuned.pkl')
print("Model and transformer saved as 'ridge_model_tuned.pkl'")
