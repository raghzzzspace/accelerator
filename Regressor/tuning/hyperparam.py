from sklearn.model_selection import GridSearchCV

def tune_model(model, param_grid, X_train, y_train, scoring='r2', cv=5):
    grid = GridSearchCV(model, param_grid, scoring=scoring, cv=cv, n_jobs =-1, verbose=1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_