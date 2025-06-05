from flask import Flask, render_template, request, send_file
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    results = None
    if request.method == 'POST':
        # Step 1: Get files and form data
        train_file = request.files['train_csv']
        test_file = request.files['test_csv']
        model_type = request.form['model']
        target_column = request.form['target']
        preprocess = request.form.getlist('preprocess')

        # Step 2: Save and read
        train_path = os.path.join(UPLOAD_FOLDER, 'train.csv')
        test_path = os.path.join(UPLOAD_FOLDER, 'test.csv')
        train_file.save(train_path)
        test_file.save(test_path)

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        # Step 3: Preprocessing
        if 'dropna' in preprocess:
            train_df.dropna(inplace=True)
            test_df.dropna(inplace=True)

        X_train = train_df.drop(columns=[target_column])
        y_train = train_df[target_column]
        X_test = test_df.copy()

        if 'normalize' in preprocess:
            scaler = MinMaxScaler()
            X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
            X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

        # Step 4: Model selection
        if model_type == 'linear':
            model = LinearRegression()
        else:
            model = RandomForestRegressor()

        # Step 5: Training and predicting
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Step 6: Evaluate on training set (not test because we don’t have y_test)
        y_train_pred = model.predict(X_train)
        mse = mean_squared_error(y_train, y_train_pred)
        r2 = r2_score(y_train, y_train_pred)

        # Step 7: Save predictions
        test_df['Predicted'] = predictions
        output_path = os.path.join(UPLOAD_FOLDER, 'predictions.csv')
        test_df.to_csv(output_path, index=False)

        results = {
            'mse': round(mse, 2),
            'r2': round(r2, 2),
            'download_link': '/download'
        }

    return render_template('index.html', results=results)

@app.route('/download')
def download_file():
    return send_file(os.path.join(UPLOAD_FOLDER, 'predictions.csv'), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
