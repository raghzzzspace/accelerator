import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from models.trainer import get_model
from utils.metrics import regression_metrics
from utils.save_model import save_model
from utils.data_split import train_test_split_data
from tuning.hyperparam import hyperparams, visualize_results, log_tuning_results
import joblib

df = pd.read_csv('C:\\Users\\Ashok Kumar\\Desktop\\iris.csv')
X = df.drop('SepalLengthCm', axis=1)
y = df['SepalLengthCm']

X_train, X_test, y_train, y_test = train_test_split_data(X, y, test_size=0.2, random_state=42)

model = get_model('linear', task='regression')

param_grid = hyperparams(str(model))

def tune_model(model, param_grid, X_train, y_train, scoring='r2', operation = 'GridSeachCV',cv=5):
    if operation == 'GridSearchCV':
        grid = GridSearchCV(model, param_grid, scoring=scoring, cv=cv, n_jobs=-1, verbose=1)
        grid.fit(X_train, y_train)
        metric_logs = []
        for i in range(len(grid.cv_results_['params'])):
            entry = {
        'params': grid.cv_results_['params'][i],
        'mean_test_score': grid.cv_results_['mean_test_score'][i],
        'mean_train_score': grid.cv_results_['mean_train_score'][i],
        'fit_time': grid.cv_results_['mean_fit_time'][i]
    }
        metric_logs.append(entry)

        return grid.best_estimator_, grid.best_params_, grid, metric_logs
    if operation == 'RandomizedSearchCV':
        random_search = RandomizedSearchCV(
            model,
            param_grid,
            n_iter=20,                
            scoring=scoring,             
            cv=cv,                     
            verbose=1,
            random_state=42,
            n_jobs=-1
            )
        random_search.fit(X_train, y_train)
        metric_logs = []
        for i in range(len(random_search.cv_results_['params'])):
            entry = {
        'mean_test_score': random_search.cv_results_['mean_test_score'][i],
        'fit_time': random_search.cv_results_['mean_fit_time'][i],
        'params': random_search.cv_results_['params'][i],
        #'mean_train_score': random_search.cv_results_['mean_train_score'][i],
    }
            metric_logs.append(entry)

        return random_search.best_estimator_, random_search.best_params_, random_search, metric_logs
    if operation == 'None':
        return model, model.get_params(), None

tuned_model, best_params, tuned_obj, metric_logs = tune_model(model, param_grid, X_train, y_train, operation = 'RandomizedSearchCV')
visualize_results(tuned_obj, param_name = 'fit_intercept')

print(log_tuning_results(tuned_model, best_params, metric_logs, save_path = 'artifacts/logs.csv'))

model = tuned_model.fit(X_train, y_train)

y_pred = model.predict(X_test)
metrics = regression_metrics(y_test, y_pred)
print("Evaluation Metrics:", metrics)

joblib.dump((tuned_model), 'artifacts/ridge_model_tuned.pkl')
print("Model and transformer saved as 'model_tuned.pkl'")

# Load the model and make predictions
#tuned_model = joblib.load('ridge_model_tuned.pkl')
#tuned_model.predict(X_test[:5])