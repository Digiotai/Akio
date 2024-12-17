from dotenv import load_dotenv
# from .database import PostgreSQLDB
import base64
import hashlib
import os
import ast
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo
import requests
import calendar
import joblib
# Import necessary libraries
import numpy as np
import pmdarima as pm
import tensorflow as tf
from digiotai.digiotai_jazz import Agent, Task, InputType, OutputType
from django.views.decorators.csrf import csrf_exempt
from dotenv import load_dotenv
from keras.models import load_model
from openai import OpenAI
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .database import PostgresDatabase, HanaDBManager

global connection_obj
# db = MongoDBDatabase()

# Configure OpenAI
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)
expertise = "Interior Designer"
task = Task("Image Generation")
input_type = InputType("Text")
output_type = OutputType("Image")
agent = Agent(expertise, task, input_type, output_type)
api_key = OPENAI_API_KEY
name = "file_name"
headers = {
    'Authorization':
        'FlespiToken flespi_token'
}

db = PostgresDatabase()

os.makedirs('uploads', exist_ok=True)


def updatedtypes(df):
    datatypes = df.dtypes
    for col in df.columns:
        if datatypes[col] == 'object':
            try:
                pd.to_datetime(df[col])
                df.drop(col, axis=1, inplace=True)
                print(df.columns)
            except Exception as e:
                pass
    return df


def get_importance(X_train, y_train, model_type):
    if model_type == 'regression':
        model_ = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        model_ = RandomForestClassifier(n_estimators=100, random_state=42)
    model_.fit(X_train, y_train)

    # Get feature importances
    feature_importance = model_.feature_importances_

    # Normalize feature importance to percentages
    feature_importance_percent = (feature_importance / np.sum(feature_importance)) * 100

    # Print feature importance scores in percentage
    df = pd.DataFrame({"Features": X_train.columns, "Importances": feature_importance_percent})
    df.sort_values(by='Importances', inplace=True, ascending=False)
    df.reset_index(inplace=True, drop=True)
    return df


def iscatcol(col, t, threshold=10):
    unique_values = col.dropna().unique()
    if len(unique_values) <= threshold or t == 'object':
        if t == 'object':
            return True, True  # Categorical and needs encoding
        return True, False  # Categorical but doesn't require encoding
    return False, False


def getcatcols(df):
    catcols = []
    catcols_encode = []
    unique_cols = {}
    for col in df.columns:
        a, b = iscatcol(df[col], df.dtypes[col])
        if a:
            catcols.append(col)
        if b:
            catcols_encode.append(col)
            unique_cols[col] = list(df[col].unique())
    return catcols, catcols_encode, unique_cols


def get_csv_metadata(df):
    metadata = {
        "columns": df.columns.tolist(),
        "data_types": df.dtypes.to_dict(),
        "null_values": df.isnull().sum().to_dict(),
        "example_data": df.head().to_dict()
    }
    return metadata


def data_cleanup(df):
    if 'Date' in df.columns and 'Time' in df.columns:
        df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
        df.drop(['Date', 'Time'], axis=1, inplace=True)  # Drop original columns if needed

    for col in df.select_dtypes(include='object').columns:
        try:
            df[col] = pd.to_datetime(df[col])
        except Exception:
            pass
    # 1. Removing columns with unique value
    for col in df.columns:
        if df[col].nunique() <= 5:
            print('dropping columns with single value', col)
            df.drop(col, axis=1, inplace=True)
    df = df.select_dtypes(include=['number', 'datetime'])
    # Selecting numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

    # 2. Dropping low variance columns
    variances = df[numeric_cols].var()
    dynamic_variance_threshold = variances.median() * 0.1  # 10% of median variance
    low_variance_numeric = variances[variances < dynamic_variance_threshold].index.tolist()

    # variance_std_dev = variances.std()
    # variance_mean = variances.mean()
    #
    # # Define threshold as mean - 1 standard deviation
    # threshold = variance_mean - variance_std_dev
    #
    # # Identify low variance columns
    # low_variance_numeric = variances[variances < threshold].index.tolist()

    print('dropping low variance columns', low_variance_numeric)
    df.drop(low_variance_numeric, axis=1, inplace=True)
    return df


@csrf_exempt
def train_data(request, train_type, file_name):
    try:
        df = pd.read_csv(os.path.join("uploads", file_name.lower() + '.csv'))
        df = df.iloc[:300, :]
        if os.path.exists(os.path.join("data", file_name.lower())):
            return HttpResponse("Success")
        if train_type.lower() == 'predict':
            df = updatedtypes(df)
            os.makedirs(os.path.join("data", file_name.lower()), exist_ok=True)
            df.to_csv(os.path.join("data", file_name.lower(), "processed_data.csv"), index=False)
            for i in df.columns:
                try:
                    col_predict = i
                    print(col_predict)
                    label_encoders = {}
                    cat_col = False
                    # Split the data into features (X) and target variable (y)
                    X = df.drop(columns=[col_predict])
                    y = df[col_predict]
                    catcols, cat_cols_to_encode, unique_cols = getcatcols(X)
                    print(catcols, cat_cols_to_encode)
                    for column in cat_cols_to_encode:
                        label_encoders[column] = LabelEncoder()
                        X[column] = label_encoders[column].fit_transform(X[column])
                    dense_c = 1
                    if iscatcol(y, y.dtype)[0]:
                        label_encoders[col_predict] = LabelEncoder()
                        y = label_encoders[col_predict].fit_transform(y)
                        dense_c = len(label_encoders[col_predict].classes_)
                        cat_col = True
                    print(dense_c, "h")
                    scaler = StandardScaler()
                    numerical_features = list(set(X.columns) - set(catcols))
                    X[numerical_features] = scaler.fit_transform(X[numerical_features])
                    print(numerical_features)
                    model_type = None
                    # Split the data into training and testing sets
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    model = tf.keras.models.Sequential([
                        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
                        tf.keras.layers.Dense(64, activation='relu')
                    ])
                    if iscatcol(df[col_predict], df.dtypes[col_predict])[0]:
                        # Define the architecture of the ANN model
                        model.add(tf.keras.layers.Dense(dense_c,
                                                        activation='softmax'))  # Output layer for binary classification
                        loss_function = 'sparse_categorical_crossentropy'
                        metrics = ['accuracy']
                        model_type = 'classification'

                    else:
                        model.add(tf.keras.layers.Dense(1))  # Output layer for regression
                        loss_function = 'mean_squared_error'
                        metrics = ['mae']
                        model_type = 'regression'
                    print(model_type)
                    # Compile the model
                    model.compile(optimizer='adam', loss=loss_function, metrics=metrics)

                    # Train the model
                    model.fit(X_train, y_train, epochs=500, batch_size=64, validation_split=0.1)

                    # Evaluate the model on the testing data
                    loss, accuracy = model.evaluate(X_test, y_test)
                    print("Test Accuracy:", accuracy)
                    if model_type == 'classification':
                        predictions = np.argmax(model.predict(X_test), axis=-1)
                    else:
                        predictions = model.predict(X_test)

                    for column in cat_cols_to_encode:
                        X_test[column] = label_encoders[column].inverse_transform(X_test[column])

                    if cat_col:
                        y_test = label_encoders[col_predict].inverse_transform(y_test)
                        predictions = label_encoders[col_predict].inverse_transform(predictions)

                    predicted_data = X_test.copy()
                    predicted_data["Actual"] = y_test
                    predicted_data["Predicted"] = predictions
                    print(predicted_data.head(5))
                    # Sort the DataFrame by importance
                    importance_df = get_importance(X_train, y_train, model_type)

                    # Print or plot the top features
                    print(importance_df)
                    if not os.path.exists(os.path.join("data", file_name.lower(), col_predict.replace(" ", "_"))):
                        os.makedirs(os.path.join("data", file_name.lower(), col_predict.replace(" ", "_")))

                    predicted_data.to_csv(
                        os.path.join("data", file_name.lower(), col_predict.replace(" ", "_"), 'predictions.csv'),
                        index=False)
                    # Save the label encoders
                    for column, encoder in label_encoders.items():
                        joblib.dump(encoder,
                                    os.path.join("data", file_name.lower(), col_predict.replace(" ", "_"),
                                                 f'{column.replace(" ", "_")}_encoder.pkl'))

                    # Save the trained model
                    print("saved_path",
                          os.path.join("data", file_name.lower(), col_predict.replace(" ", "_"), "model.h5"))
                    model.save(os.path.join("data", file_name.lower(), col_predict.replace(" ", "_"), "model.h5"))
                    if model_type == 'classification':
                        accuracy = accuracy * 100
                        metrics = "Accuracy"
                    else:
                        metrics = "MAE"

                    results = {
                        "accuracy": accuracy,
                        "metrics": metrics,
                        "Top Fields": importance_df.to_json(),
                        "Plot": [int(value) for value in X_train[importance_df[importance_df.columns[0]][0]].values],
                        "sample_rows": predicted_data.to_json()
                    }
                    cols = {c: unique_cols[c] if c in unique_cols else None for c in X_train.columns}

                    with open(os.path.join("data", file_name.lower(), col_predict.replace(" ", "_"), "deployment.json"),
                              "w") as fp:
                        json.dump({"columns": cols, "model_type": model_type}, fp, indent=4)
                    with open(os.path.join("data", file_name.lower(), col_predict.replace(" ", "_"), "results.json"),
                              "w") as fp:
                        json.dump(results, fp, indent=4)
                except Exception as e:
                    print(e)
                    return HttpResponse("Error " + str(e))
            return HttpResponse('Success')
        elif train_type.lower() == 'forecast':
            df = data_cleanup(df)
            data = df
            try:
                numeric_cols = df.select_dtypes(include=['datetime']).columns.tolist()
                date_column = numeric_cols[0]
                if not date_column:
                    raise ValueError("No datetime column found in the dataset.")
                print(date_column)
                # Set the date column as index
                data[date_column] = pd.to_datetime(data[date_column])
                data.set_index(date_column, inplace=True)
                print(data.head(15))
                # Identify forecast columns (numeric columns)
                forecast_columns = data.select_dtypes(include=[np.number]).columns.tolist()
                if not forecast_columns:
                    raise ValueError("No numeric columns found for forecasting in the dataset.")

                time_differences = data.index.to_series().diff().dropna()
                print(time_differences)
                inconsistent_intervals = time_differences[time_differences != time_differences.mode()[0]]
                print(inconsistent_intervals)

                # Infer frequency of datetime index
                freq = pd.infer_freq(data.index)
                print(date_column, freq)
                # Determine m based on inferred frequency
                if freq == '5T':  # Five-minute data
                    m = 288  # Daily seasonality (288 intervals in a day)
                elif freq == '15T':  # Quarter-hourly data (every 15 minutes)
                    m = 96  # Daily seasonality (96 intervals in a day)
                elif freq == '30T':  # Half-hourly data (every 30 minutes)
                    m = 48  # Daily seasonality (48 intervals in a day)
                elif freq == 'H':  # Hourly data
                    m = 24  # Daily seasonality (24 intervals in a day)
                elif freq == 'D':  # Daily data
                    m = 7  # Weekly seasonality (7 days in a week)
                elif freq == 'W':  # Weekly data
                    m = 52  # Yearly seasonality (52 weeks in a year)
                elif freq == 'M':  # Monthly data
                    m = 12  # Yearly seasonality (12 months in a year)
                elif freq == 'Q':  # Quarterly data
                    m = 4  # Yearly seasonality (4 quarters in a year)
                elif freq == 'A':  # Annual data
                    m = 1  # No further seasonality within a year
                else:
                    raise ValueError(
                        f"Unsupported frequency '{freq}'. Ensure data is in a common time interval.")

                results = {}
                for col in forecast_columns:
                    try:
                        data_actual = data[col].dropna()  # Remove NaNs if any

                        # Split data into train and test sets
                        train = data_actual.iloc[:-m]
                        test = data_actual.iloc[-m:]

                        # Auto ARIMA model selection
                        model = pm.auto_arima(train,
                                              m=m,  # frequency of seasonality
                                              seasonal=True,  # Enable seasonal ARIMA
                                              d=None,  # determine differencing
                                              test='adf',  # adf test for differencing
                                              start_p=0, start_q=0,
                                              max_p=12, max_q=12,
                                              D=None,  # let model determine seasonal differencing
                                              trace=True,
                                              error_action='ignore',
                                              suppress_warnings=True,
                                              stepwise=True)

                        # Forecast and calculate errors
                        fc, confint = model.predict(n_periods=m, return_conf_int=True)
                        # Save results to dictionary
                        results = {
                            "actual": {
                                "date": list(test.index.astype(str)),
                                "values": [float(val) if isinstance(val, np.float_) else int(val) for val in
                                           test.values]
                            },
                            "forecast": {
                                "date": list(test.index.astype(str)),
                                "values": [float(val) if isinstance(val, np.float_) else int(val) for val in fc]
                            }
                        }
                        if not os.path.exists(os.path.join("data", file_name.lower(), col.replace(" ", "_"))):
                            os.makedirs(os.path.join("data", file_name.lower(), col.replace(" ", "_")), exist_ok=True)
                        col = col.replace(" ", "_")
                        with open(os.path.join('data', file_name.lower(), col,
                                               col.lower() + '_results.json'), 'w') as fp:
                            json.dump(results, fp)
                        print(
                            f"Results saved to {os.path.join('data', file_name.lower(), col, col.lower() + '_results.json')}")
                    except Exception as e:
                        print(e)
                        return HttpResponse("Error " + str(e))
                return HttpResponse("Success")
            except Exception as e:
                print(e)
                return HttpResponse("Error " + str(e))
    except Exception as e:
        print(e)
        return HttpResponse("Error " + str(e))


# Database connection
@csrf_exempt
def connection(request):
    global connection_obj
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        database = request.POST['database']
        host = request.POST['host']
        port = request.POST['port']
        connection_obj = db.create_connection(username, password, database, host, port)
        print(connection_obj)
        return HttpResponse(json.dumps({"tables": connection_obj}), content_type="application/json")


# Upload data to the database
# Upload data to the database (CSV and Excel)
import os
import io
import shutil
import json
import pandas as pd
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
import xmltodict
from django.core.cache import cache
from django.core.exceptions import SuspiciousOperation


@csrf_exempt
def upload_and_analyze_data(request):
    try:
        if request.method == 'POST':
            email = request.POST.get('mail')
            files = request.FILES.get('file')  # Retrieve the uploaded file
            kpi_file = request.FILES.get("kpi_file")

            if not files:
                return JsonResponse({"error": "No files uploaded"}, status=400)

            file_name = files.name
            file_extension = os.path.splitext(file_name)[1].lower()  # Extract file extension

            try:
                # Process the uploaded file based on its extension
                if file_extension == '.csv':
                    content = files.read().decode('utf-8')
                    csv_data = io.StringIO(content)
                    df = pd.read_csv(csv_data)
                elif file_extension in ['.xls', '.xlsx']:
                    df = pd.read_excel(files)
                else:
                    raise SuspiciousOperation("Unsupported file format")

                # Save the uploaded file locally for backup/logging purposes
                upload_dir = "uploads"
                os.makedirs(upload_dir, exist_ok=True)

                csv_file_path = os.path.join(upload_dir, file_name.replace(file_extension, '.csv').lower())
                df.to_csv(csv_file_path, index=False)

                excel_file_path = os.path.join(upload_dir, file_name.replace(file_extension, '.xlsx').lower())
                df.to_excel(excel_file_path, index=False, engine='openpyxl')

                df.to_csv('data.csv', index=False)
                df.to_excel('data1.xlsx', index=False, engine='openpyxl')

                results = db.insert_or_update(email, df, file_name)  # Insert into MongoDB

                data_file_name, kpi_config_file_name = file_name, ''

                new_df, html_df = process_missing_data(df.copy())
                cache.set('dataframe', html_df)
                request.session['dataframe'] = html_df

                new_df.to_csv(os.path.join('uploads', 'processed_data.csv'), index=False)

                if os.path.exists('kpis.json'):
                    os.remove('kpis.json')
                request.session['uploadedFileName'] = files.name

                if kpi_file:
                    kpis_dict = xmltodict.parse(kpi_file.read())
                    with open('uploads/kpi_config.json', 'w') as json_file:
                        json.dump(kpis_dict, json_file, indent=4)
                        kpi_config_file_name = kpi_file.name

                with open('uploads/configs.json', 'w') as json_file:
                    json.dump({
                        "data_file_name": data_file_name,
                        "kpi_config_file_name": kpi_config_file_name
                    }, json_file, indent=4)

                response_data1 = analyze_data(df)  # Assuming analyze_data is a function that analyzes data
                response_data1['preview'] = df.head(10).to_dict(orient='records')
                response_data1['upload_status'] = results

                return JsonResponse(response_data1, safe=False)

            except Exception as e:
                return JsonResponse({"error": f"Failed to upload and analyze file: {str(e)}"}, status=500)

        elif request.method == 'GET':
            if os.path.exists('uploads') and os.path.exists(os.path.join('uploads', 'data.csv')):
                data_frame = pd.read_csv(os.path.join('uploads', 'data.csv'))
                data_frame = updatedtypes(data_frame)

                with open('uploads/configs.json', 'r') as json_file:
                    data = json.load(json_file)
                    return JsonResponse({
                        "uploadedInfo": True,
                        "df_preview": data_frame.head().to_dict(orient='records'),
                        "data_file_name": data['data_file_name'],
                        "kpi_config_file_name": data["kpi_config_file_name"]
                    })

            return JsonResponse({"uploadedInfo": False})

        return JsonResponse({"error": "Invalid Request Method"}, status=405)

    except Exception as e:
        return JsonResponse({"error": f"An error occurred: {str(e)}"}, status=500)


def analyze_data(df):
    # Extract the first 10 rows of the data
    first_10_rows = df.head(10).to_dict(orient='records')
    # Generate descriptions and questions for the data based on table name and columns
    columns_list = ", ".join(df.columns)
    text_questions = {}
    plotting_questions = {}
    forecasting_questions = {}
    prompt_eng = (
        f"You are analytics_bot. Analyse the data: {df.head()} and give description of the columns"
    )
    column_description = generate_code(prompt_eng)
    trials = 3
    while trials > 0:
        try:
            prompt_eng1 = (
                    f"Based on the data with sample records as {df.head()}, generate 5 questions based on data." + "output should be in the format like  {'question1':...., 'question2':...., so on..} ")
            text_questions = generate_code(prompt_eng1)
            text_questions = ast.literal_eval(text_questions)
            break
        except Exception as e:
            print(e)
        trials -= 1
    trials = 3
    while trials > 0:
        try:
            prompt_eng_2 = f"Based on the data with sample records as {df.head()}, " + "Generate 5 plotting questions based on data, Each question should start with plot keyword. output should be in the format like  {'question1':...., 'question2':...., so on..}"
            plotting_questions = generate_code(prompt_eng_2)
            plotting_questions = ast.literal_eval(plotting_questions)
            break
        except Exception as e:
            print(e)
        trials -= 1
    trials = 3
    while trials > 0:
        try:
            # Creating the forecasting questions
            prompt_eng_3 = (
                # f"Generate 5 forecasting questions for the data: {df}"

                f"Using the dataset {df}, generate 5 forecasting-related questions based on the dataset. "
                f"The questions should: "
                f"1. Start with the word **'Forecast'**. "
                f"2. The questions should be very simple and straight forward."
                f"3. Design the questions to give visually interpretable outputs, such as charts or graphs, for forecasting analysis."
                f"4. Examples: 'Forecast the sales trend for the next 6 months' or 'Forecast the quarterly revenue growth for the next year.'"
                "5. output should be in the format like  {'question1':...., 'question2':...., so on..} "

                # f"Given the dataset {df}, generate 5 forecasting-related questions that meet the following criteria: "
                # f"1. The questions should be **specific**, **realistic**, and **focused on measurable metrics or trends**. "
                # f"2. Each question should **start with 'Forecast'** and address **clear forecasting goals**. "
                # f"3. Tailor the questions to the type of data in the dataset, ensuring they align with the trends, patterns, or key variables observed in the data."

                # f"Using the dataset {df}, generate 5  forecasting related questions. "
                # f"Each question must start with 'Forecast' and focus on predicting measurable trends within the dataset. "
                # f"Design the questions to give visually interpretable outputs, such as charts or graphs, for forecasting analysis. "
                # f"Examples: 'Forecast the sales trend for the next 6 months' or 'Forecast the quarterly revenue growth for the next year.' "
            )

            forecasting_questions = generate_code(prompt_eng_3)

            forecasting_questions = ast.literal_eval(forecasting_questions)
            break
        except Exception as e:
            print(e)
        trials -= 1
    # Create a JSON response with titles corresponding to each prompt
    response_data = {
        "all_records": df.to_dict(orient='records'),
        "first_10_rows": first_10_rows,  # Include first 10 rows
        "column_description": column_description,
        "text_questions": text_questions,
        "plotting_questions": plotting_questions,
        "forecasting_questions": forecasting_questions
    }
    return response_data


def process_missing_data(df):
    df = convert_to_datetime(df)
    df, html_df = handle_missing_data(df)
    return df, html_df


def convert_to_datetime(df):
    # Define the possible date formats to try
    date_formats = ['%m-%d-%Y', '%m/%d/%Y', '%d-%m-%Y', '%d/%m/%Y', '%Y-%m-%d', '%Y/%m/%d']

    # Loop through each column
    for col in df.columns:
        # Only process object columns, assuming they may contain dates in string format
        if df[col].dtype == 'object':
            # Check if the column contains potential date strings
            if df[col].str.contains(r'\d{1,4}[-/]\d{1,2}[-/]\d{1,4}', na=False).any():
                # Try to parse automatically first
                try:
                    df[col] = pd.to_datetime(df[col], errors='raise')
                except (ValueError, TypeError):
                    # If automatic parsing fails, try each format individually
                    def parse_date(value):
                        for fmt in date_formats:
                            try:
                                return pd.to_datetime(value, format=fmt)
                            except (ValueError, TypeError):
                                continue
                        return pd.NaT  # Return NaT if none of the formats match

                    # Apply the custom parse function to handle multiple formats
                    df[col] = df[col].apply(parse_date)
    return df


from sklearn.impute import KNNImputer


def handle_missing_data(df):
    try:
        # Identify numeric and datetime columns
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        date_time_cols = df.select_dtypes(include=['datetime64']).columns

        # Impute numeric columns and track which cells were imputed
        imputer = KNNImputer(n_neighbors=5)
        imputed_numeric = imputer.fit_transform(df[numeric_cols])
        imputed_numeric_df = pd.DataFrame(imputed_numeric, columns=numeric_cols)

        # Mark imputed cells (True if the original cell was NaN)
        imputed_flags = df[numeric_cols].isnull()
        imputed_flags = imputed_flags.applymap(lambda x: x if x else False)

        # Update DataFrame with imputed values
        df[numeric_cols] = imputed_numeric_df

        # Handle datetime columns by forward filling missing values
        for col in date_time_cols:
            df[col] = pd.to_datetime(df[col])
            time_diffs = df[col].diff().dropna()
            avg_diff_sec = time_diffs.mean().total_seconds()
            minute_sec = 60
            hour_sec = 3600
            day_sec = 86400
            month_sec = day_sec * 30.44
            year_sec = day_sec * 365.25

            if avg_diff_sec < hour_sec:
                time_unit = "minutes"
                avg_diff = pd.Timedelta(minutes=avg_diff_sec / minute_sec)
            elif avg_diff_sec < day_sec:
                time_unit = "hours"
                avg_diff = pd.Timedelta(hours=avg_diff_sec / hour_sec)
            elif avg_diff_sec < month_sec:
                time_unit = "days"
                avg_diff = pd.Timedelta(days=avg_diff_sec / day_sec)
            elif avg_diff_sec < year_sec:
                time_unit = "months"
                avg_diff = pd.DateOffset(months=round(avg_diff_sec / month_sec))
            else:
                time_unit = "years"
                avg_diff = pd.DateOffset(years=round(avg_diff_sec / year_sec))

            for i in range(1, len(df)):
                if pd.isnull(df[col].iloc[i]):
                    df.loc[i, col] = df[col].iloc[i - 1] + avg_diff
                    imputed_flags.loc[i, col] = True

            imputed_flags.fillna(False, inplace=True)

        # Convert the DataFrame into a JSON-serializable format with flags
        data = []
        for _, row in df.iterrows():
            row_data = {}
            for col in df.columns:
                row_data[col] = {
                    "value": row[col].strftime('%Y-%m-%d %H:%M:%S') if isinstance(row[col], pd.Timestamp) else row[col],
                    "is_imputed": str(imputed_flags[col].get(_, False)) if col in imputed_flags else str(False)
                    # Check if cell was imputed
                }
            data.append(row_data)
        return df, data
    except Exception as e:
        print(e)


def serialize_datetime(obj):
    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    raise TypeError("Type not serializable")


# Showing the number of tables in the database
@csrf_exempt
def get_tableinfo(request):
    if request.method == 'POST':
        table_info = db.get_tables_info()
        return HttpResponse(table_info, content_type="application/json")


@csrf_exempt
def get_user_data(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        table_info = db.get_user_tables(email)
        print(table_info)
        return HttpResponse(json.dumps({"result": table_info}), content_type="application/json")


# Showing the data to the user based on the table name
@csrf_exempt
def read_db_table_data(request):
    if request.method == 'POST':
        tablename = request.POST['tablename']
        df = db.get_table_data(tablename)
        df.to_csv('data.csv', index=False)
        df.to_csv(os.path.join("uploads", tablename.lower() + '.csv'), index=False)
        response_data = analyze_data(df)
        return JsonResponse(response_data, safe=False)
        # print(response_data)
        # return HttpResponse(json.dumps({"result": response_data}, default=serialize_datetime),
        #                     content_type="application/json")


@csrf_exempt
def read_data(request):
    if request.method == 'POST':
        tablename = request.POST['tablename']
        df = db.get_table_data(tablename)
        df.to_csv('data.csv', index=False)
        df.to_csv(os.path.join("uploads", tablename.lower() + '.csv'), index=False)
        return HttpResponse(df.to_json(), content_type="application/json")


# Function to generate code from OpenAI API
def generate_code(prompt_eng):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_eng}
        ]
    )
    all_text = ""
    for choice in response.choices:
        message = choice.message
        chunk_message = message.content if message else ''
        all_text += chunk_message
    print(all_text)
    if "```python" in all_text:
        code_start = all_text.find("```python") + 9
        code_end = all_text.find("```", code_start)
        code = all_text[code_start:code_end]
    else:
        code = all_text
    return code


@csrf_exempt
def regenerate_txt(request):
    if request.method == "POST":
        df = pd.read_csv('data.csv')
        prompt_eng = (
                f"Based on the data with sample records as {df.head()}, generate 5 questions based on data." + " output should be in the format like  {'question1':...., 'question2':...., so on..}"
        )
        text_questions = {}
        trials = 3
        while trials > 0:
            try:
                text_questions = generate_code(prompt_eng)
                text_questions = ast.literal_eval(text_questions)
                break
            except Exception as e:
                print(e)
            trials -= 1
        return HttpResponse(json.dumps({"questions": text_questions}),
                            content_type="application/json")


@csrf_exempt
def regenerate_chart(request):
    if request.method == "POST":
        df = pd.read_csv('data.csv')
        prompt_eng = (
                f"Based on the data with sample records as {df.head()}. Generate 5 plotting questions based on data. question shoud start with plot keyword" + " output should be in the format like  {'question1':...., 'question2':...., so on..}"
        )
        code = {}
        trials = 3
        while trials > 0:
            try:
                code = generate_code(prompt_eng)
                code = ast.literal_eval(code)
                break
            except Exception as e:
                print(e)
            trials -= 1
        return HttpResponse(json.dumps({"questions": code}),
                            content_type="application/json")


# For Forecast
@csrf_exempt
def regenerate_forecast(request):
    if request.method == "POST":
        df = pd.read_csv('data.csv')
        prompt_eng = (
                f"Using the dataset {df}, Regenerate 5 forecasting-related questions based on the dataset. "
                f"The questions should: "
                f"1. Start with the word **'Forecast'**. "
                f"2. The questions should be very simple and straight forward."
                f"3. Design the questions to give visually interpretable outputs, such as charts or graphs, for forecasting analysis."
                f"4. Examples: 'Forecast the sales trend for the next 6 months' or 'Forecast the quarterly revenue growth for the next year.'"
                + "5. output should be in the format like  {'question1':...., 'question2':...., so on..}"
        )
        trials = 3
        code = {}
        while trials > 0:
            try:
                code = generate_code(prompt_eng)
                code = ast.literal_eval(code)
                break
            except Exception as e:
                print(e)
            trials -= 1
        return HttpResponse(json.dumps({"questions": code}),
                            content_type="application/json")


@csrf_exempt
def gen_txt_response(request):
    if request.method == "POST":
        csv_file_path = 'data.csv'
        df = pd.read_csv(csv_file_path)
        # Generate CSV metadata
        csv_metadata = {"columns": df.columns.tolist()}
        metadata_str = ", ".join(csv_metadata["columns"])
        query = request.POST["query"]
        prompt_eng = (
            f"""
                You are a Python expert focused on answering user queries about data preprocessing. Always strictly adhere to the following rules:               
                1. Generic Queries:
                    If the user's query is generic and not related to data, respond with a concise and appropriate print statement. For example:

                    Query: "What is AI?"
                    Response: "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines."
                2. Data-Related Queries:
                    If the query is about data processing, assume the file data.csv is the data source and contains the following columns: {metadata_str}.

                    For these queries, respond with Python code only, no additional explanations.
                    The code should:

                    Load data.csv using pandas.
                    Perform operations to directly address the query.
                    Exclude plotting, visualization, or other unnecessary steps.
                    Include comments for key steps in the code.
                    Example:

                    Query: "How can I filter rows where 'Column1' > 100?"
                    Response:
                    python
                    Copy code
                    import pandas as pd

                    # Load the dataset
                    data = pd.read_csv('data.csv')

                    # Filter rows where 'Column1' > 100
                    filtered_data = data[data['Column1'] > 100]

                    # Output the result
                    print(filtered_data)

                3. Theoretical Concepts:
                    For theoretical questions, provide a brief explanation as a print statement. Keep the explanation concise and focused.

                    Example:

                    Query: "What is normalization in data preprocessing?"
                    Response:
                    "Normalization is a data preprocessing technique used to scale numeric data within a specific range, typically [0, 1], to ensure all features contribute equally to the model."

                Never reply with: "Understood!" or similar confirmations. Always directly respond to the query following the above rules.

                User query is {query}.
            """
        )
        code = generate_code(prompt_eng)
        # Execute the generated code
        result = execute_py_code(code, df)
        return JsonResponse({"answer": result})
    return HttpResponse("Invalid Request Method", status=405)


def execute_py_code(code, df):
    # Create a string buffer to capture the output
    buffer = io.StringIO()
    sys.stdout = buffer

    # Create a local namespace for execution
    local_vars = {'df': df}

    try:
        # Execute the code
        exec(code, globals(), local_vars)

        # Get the captured output
        output = buffer.getvalue().strip()

        # If there's no output, try to get the last evaluated expression
        if not output:
            last_line = code.strip().split('\n')[-1]
            if not last_line.startswith(('print', 'return')):
                output = eval(last_line, globals(), local_vars)
                print(output)
    except Exception as e:
        output = f"Error executing code: {str(e)}"
    finally:
        # Reset stdout
        sys.stdout = sys.__stdout__

    return str(output)


# For Genai
from django.views.decorators.csrf import csrf_exempt


@csrf_exempt
def gen_graph_response(request):
    if request.method == "POST":

        csv_file_path = 'data.csv'
        df = pd.read_csv(csv_file_path)

        # Generate CSV metadata
        csv_metadata = {"columns": df.columns.tolist()}
        metadata_str = ", ".join(csv_metadata["columns"])

        query = request.POST["query"]

        prompt_eng = (
            f"You are an AI specialized in data analytics and visualization."
            f" Data used for analysis is stored in a CSV file data.csv."
            f"Attributes of the data are: {metadata_str}."
            f"Consider 'data.csv' as the data source for any analysis."
            f"Based on the query generate only the Python code using Matplotlib to plot the graph."
            f"Save the graph as 'graph.png'. Also save the graph description in description.txt"
            f"The user asks: {query}"
        )

        code = generate_code(prompt_eng)
        print(code)

        if 'import matplotlib' in code:
            try:
                exec(code)
                # Send the generated 'graph.png' as the file response
                image_path = "graph.png"
                if os.path.exists(image_path):
                    return FileResponse(open(image_path, 'rb'), content_type='image/png')
                else:
                    return HttpResponse("Graph image not found", status=404)
            except Exception as e:
                prompt_eng = f"There has occurred an error while executing the code, please take a look at the error " \
                             f"and strictly only reply with the full python code. Do not apologize or anything; just " \
                             f"give the code. {str(e)}"
                code = generate_code(prompt_eng)
                try:
                    exec(code)
                    # Send the generated 'graph.png' as the file response
                    image_path = "graph.png"
                    if os.path.exists(image_path):
                        return FileResponse(open(image_path, 'rb'), content_type='image/png')
                    else:
                        return HttpResponse("Graph image not found", status=404)
                except Exception as e:
                    return HttpResponse("Failed to generate the chart. Please try again")
        else:
            return HttpResponse(code)


@csrf_exempt
def get_description(request):
    try:
        with open('description.txt', 'r') as fp:
            data = fp.read()
        return HttpResponse(json.dumps({"description": data}), content_type="application/json")
    except Exception as e:
        print(e)


# For genbi
from django.views.decorators.csrf import csrf_exempt
import io
import sys


@csrf_exempt
def genresponse2(request):
    if request.method == "POST":
        df = pd.read_csv('data.csv')

        # Generate CSV metadata
        csv_metadata = {"columns": df.columns.tolist()}
        metadata_str = ", ".join(csv_metadata["columns"])

        query = request.POST["query"]

        print("execution started")

        prompt_eng = (
            f"You are an AI specialized in data preprocessing."
            f"Data related to the {query} is stored in a CSV file data.csv. Consider the data.csv as the data source"
            f"Generate Python code to answer the question: {query}."
            f"The data contains the following columns: {metadata_str}. "
            f"Return only the Python code that computes the result .Result should describe the parameters in it, "
            f"without any plotting or visualization."
            f"If the {query} related to the theoretical concept.You will give a small description about the concept also."
        )

        code = generate_code(prompt_eng)

        print(code)

        # Execute the generated code
        result = execute_py_code(code, df)

        return JsonResponse({"answer": result})

    return HttpResponse("Invalid Request Method", status=405)


def execute_py_code(code, df):
    # Create a string buffer to capture the output
    buffer = io.StringIO()
    sys.stdout = buffer

    # Create a local namespace for execution
    local_vars = {'df': df}

    try:
        # Execute the code
        exec(code, globals(), local_vars)

        # Get the captured output
        output = buffer.getvalue().strip()

        # If there's no output, try to get the last evaluated expression
        if not output:
            last_line = code.strip().split('\n')[-1]
            if not last_line.startswith(('print', 'return')):
                output = eval(last_line, globals(), local_vars)
                print(output)
    except Exception as e:
        output = f"Error executing code: {str(e)}"
    finally:
        # Reset stdout
        sys.stdout = sys.__stdout__

    return str(output)


# For Genbi
from django.http import HttpResponse, FileResponse
from django.views.decorators.csrf import csrf_exempt


@csrf_exempt
def genAIPrompt2(request):
    if request.method == "POST":
        df = pd.read_csv("data.csv")

        # Generate CSV metadata
        csv_metadata = {"columns": df.columns.tolist()}
        metadata_str = ", ".join(csv_metadata["columns"])

        query = request.POST["query"]

        prompt_eng = (
            f"You are an AI specialized in data analytics and visualization. "
            f" Data used for analysis is stored in a CSV file data.csv. "
            f"Attributes of the data are: {metadata_str}. "
            f"Consider 'data.csv' as the data source for any analysis."
            f"If the user asks for a graph, generate only the Python code using Matplotlib to plot the graph. "
            f"Save the graph as 'graph.png'. Also describe the graph and store the description in description.txt"
            f"If the user does not ask for a graph, simply answer the query with the computed result. "
            f"The user asks: {query}"
        )

        code = generate_code(prompt_eng)
        print(code)

        if 'import matplotlib' in code:
            try:
                exec(code)
                # Send the generated 'graph.png' as the file response
                image_path = "graph.png"
                if os.path.exists(image_path):
                    return FileResponse(open(image_path, 'rb'), content_type='image/png')
                else:
                    return HttpResponse("Graph image not found", status=404)
            except Exception as e:
                prompt_eng = f"There has occurred an error while executing the code, please take a look at the error and strictly only reply with the full python code. Do not apologize or anything; just give the code. {str(e)}"
                code = generate_code(prompt_eng)
                try:
                    exec(code)
                    # Send the generated 'graph.png' as the file response
                    image_path = "graph.png"
                    if os.path.exists(image_path):
                        return FileResponse(open(image_path, 'rb'), content_type='image/png')
                    else:
                        return HttpResponse("Graph image not found", status=404)
                except Exception as e:
                    return HttpResponse("Failed to generate the chart. Please try again")
        else:
            return HttpResponse(code)


# Predict and forecast purpose
# Getting the prediction result
@csrf_exempt
def get_prediction_info(request, data, field):
    with open(f"data/{data.lower()}/{field}/results.json", 'r') as fp:
        res = json.load(fp)
    return HttpResponse(json.dumps({"data": res}), content_type="application/json")


# This will return the columns for the table
@csrf_exempt
def get_columns(request, train_type, data):
    if train_type == 'predict':
        df = pd.read_csv(f"data/{data.lower()}/processed_data.csv")
        cols = set(df.columns) - {"Store ID", "Employee Number", "Area"}
        return HttpResponse(json.dumps({"columns": list(cols)}), content_type="application/json")
    elif train_type == 'forecast':
        cols = os.listdir(f"data/{data.lower()}")
        return HttpResponse(json.dumps({"columns": list(cols)}), content_type="application/json")
    else:
        return HttpResponse(json.dumps({"columns": []}), content_type="application/json")


# Generating the URL for the prediction
@csrf_exempt
def generate_deployment(request, data, field):
    hash_object = hashlib.sha256(f'{data}__{field}'.encode('ascii'))
    hex_dig = hash_object.hexdigest()
    save_deployments(hex_dig, data, field)
    return HttpResponse(json.dumps({"deployment_url": hex_dig}), content_type="application/json")


# THis will be helpful for the prediction on which columns.
@csrf_exempt
def deployment(request, data):
    model = get_deployment_txt(data)
    data, field = model.split('___')
    with open(f"data/{data.lower()}/{field.replace(' ', '_')}/deployment.json", 'r') as fp:
        data = json.load(fp)
    return HttpResponse(json.dumps({"columns": data["columns"]}), content_type="application/json")


# For prediction based on the deployment url
@csrf_exempt
def deployment_predict(request, data):
    if request.method == 'POST':
        model = get_deployment_txt(data)
        data, field = model.split('___')
        res = {}
        for col in request.POST:
            res.update({col: request.POST[col]})
        df = pd.DataFrame([res])
        result = load_models(data, field, df)
        if isinstance(result, np.ndarray):
            result = str(result[0])
        return HttpResponse(json.dumps({"result": str(result)}), content_type="application/json")


# Forecast of the data with the help of the deployment url
@csrf_exempt
def deployment_forecast(request, data, col):
    if request.method == 'POST':
        msg = 'Add Logic here for forecast'
        return HttpResponse(json.dumps({"result": msg, "msg": msg}), content_type="application/json")


def load_models(path, prediction_col, df):
    try:
        model = load_model(os.path.join('data', path.lower(), prediction_col.replace(" ", "_"), "model.h5"))
        with open(os.path.join('data', path.lower(), prediction_col.replace(" ", "_"), "deployment.json"), 'r') as fp:
            deployment_data = json.load(fp)
        for column in deployment_data["columns"]:
            if isinstance(deployment_data["columns"][column], list):
                encoder_path = os.path.join('data', path.lower(), prediction_col.replace(" ", "_"),
                                            f'{column.replace(" ", "_")}_encoder.pkl')
                df[column.replace("_", " ")] = joblib.load(encoder_path).fit_transform(df[column.replace("_", " ")])
            else:
                df[column] = df[column].astype(float)
        res = model.predict(df.iloc[0, :].to_numpy().reshape(1, -1))
        model_type = deployment_data["model_type"]
        if model_type == 'classification':
            result = np.argmax(res, axis=-1)
            res = joblib.load(
                os.path.join('data', path.lower(), prediction_col.replace(" ", "_"),
                             f'{prediction_col.replace(" ", "_")}_encoder.pkl')).inverse_transform(
                result)
        return res[0]
    except Exception as e:
        print(e)


def save_deployments(hex_data, data, field):
    if not os.path.exists("deployments.json"):
        deployment_data = {}
    else:
        with open("deployments.json", 'r') as fp:
            deployment_data = json.load(fp)
    deployment_data.update({hex_data: f'{data}___{field.replace(" ", "_")}'})
    with open("deployments.json", 'w') as fp:
        json.dump(deployment_data, fp)


def get_deployment_txt(hex_data):
    with open("deployments.json", 'r') as fp:
        deployment_data = json.load(fp)
    return deployment_data[hex_data]


# For Forecasting using module:
from wyge.models.openai import ChatOpenAI
from wyge.agents.react_agent import Agent
from wyge.tools.prebuilt_tools import execute_query, execute_code, install_library
from wyge.tools.raw_functions import file_to_sql, get_metadata
from .system_prompt3 import forecasting_prompt
from datetime import datetime
from django.conf import settings


def delete_images_in_current_directory() -> None:
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']

    current_directory = os.getcwd()

    for filename in os.listdir(current_directory):

        _, extension = os.path.splitext(filename)

        if extension.lower() in image_extensions:
            file_path = os.path.join(current_directory, filename)
            try:
                os.remove(file_path)
            except OSError as e:
                print(f"Error: {e} - {file_path}")


def get_images_in_directory(directory):
    """
    Fetches all image files from the specified directory.

    Parameters:
    - directory: Directory to search for image files.

    Returns:
    - List of image file paths.
    """
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
    image_files = []

    for filename in os.listdir(directory):
        _, extension = os.path.splitext(filename)
        if extension.lower() in image_extensions:
            image_files.append(os.path.join(directory, filename))

    return image_files


# Shared database credentials
USER = 'test_owner'
PASSWORD = 'tcWI7unQ6REA'
HOST = 'ep-yellow-recipe-a5fny139.us-east-2.aws.neon.tech:5432'
DATABASE = 'test'

import time


def handle_forecasting(df, openai_api_key, user_prompt, table_name="default_table"):
    """
    Processes a DataFrame for forecasting tasks by storing it in the database,
    converting it to SQL, and generating a forecast based on the user query.

    Parameters:
    - df: DataFrame to be processed.
    - openai_api_key: OpenAI API key for model access.
    - user_prompt: Query to be processed by the AI agent.

    Returns:
    - Dictionary with forecast results and optionally an image path.
    """
    print("[DEBUG] Entering handle_forecasting function")
    start_time = time.time()

    # Save the DataFrame to a temporary CSV file
    try:
        csv_file_path = os.path.join(settings.MEDIA_ROOT, f"{table_name}.csv")
        print(f"[DEBUG] Saving DataFrame to CSV file at: {csv_file_path}")
        csv_save_start = time.time()
        df.to_csv(csv_file_path, index=False)
        print(f"[DEBUG] CSV saved in {time.time() - csv_save_start:.2f} seconds")
    except Exception as e:
        print(f"[ERROR] Failed to save DataFrame to CSV: {e}")
        return JsonResponse({"error": "Failed to save DataFrame to CSV file."}, status=500)

    # Convert the CSV file to an SQL table
    try:
        print(f"[DEBUG] Converting CSV file to SQL table: {table_name}")
        sql_conversion_start = time.time()
        file_to_sql(csv_file_path, table_name, USER, PASSWORD, HOST, DATABASE)
        print(f"[DEBUG] CSV converted to SQL table in {time.time() - sql_conversion_start:.2f} seconds")
    except Exception as e:
        print(f"[ERROR] Error converting CSV to SQL table: {e}")
        return JsonResponse({"error": "Failed to convert CSV file to SQL table."}, status=500)

    # Initialize forecasting tools and LLM
    print("[DEBUG] Initializing forecasting tools and LLM")
    tools = [execute_query(), execute_code(), install_library()]
    llm = ChatOpenAI(memory=True, tools=tools, api_key=openai_api_key)

    # Retrieve metadata for the database tables
    try:
        print(f"[DEBUG] Retrieving metadata for table: {table_name}")
        metadata_start = time.time()
        metadata = get_metadata(HOST, USER, PASSWORD, DATABASE, [table_name])
        print(f"[DEBUG] Metadata retrieved in {time.time() - metadata_start:.2f} seconds")
    except Exception as e:
        print(f"[ERROR] Failed to retrieve metadata: {e}")
        return JsonResponse({"error": "Failed to retrieve metadata for the database."}, status=500)

    # Initialize the AI agent with the forecasting prompt
    print("[DEBUG] Initializing AI agent with forecasting prompt")
    agent = Agent(llm, react_prompt=forecasting_prompt)

    # Clean up any previous generated images
    print("[DEBUG] Deleting previous images from the directory")
    image_cleanup_start = time.time()
    delete_images_in_current_directory()
    print(f"[DEBUG] Previous images deleted in {time.time() - image_cleanup_start:.2f} seconds")

    # Prepare the forecasting command
    command = f"""
        Answer the user query from the database below, also use the provided tools.
        user = '{USER}'
        password = '{PASSWORD}'
        host = '{HOST}'
        database = '{DATABASE}'
        tables related to user are: [{table_name}]
        Metadata of the tables: {metadata}
        User query: {user_prompt}
    """
    print(f"[DEBUG] Command prepared for agent: {command}")

    # Execute the command and fetch the result
    try:
        print("[DEBUG] Executing command with AI agent")
        ai_execution_start = time.time()
        response = agent(command)
        response = response.split('Answer:')[-1]
        print(f"[DEBUG] AI response fetched in {time.time() - ai_execution_start:.2f} seconds")
    except Exception as e:
        print(f"[ERROR] Failed to execute forecasting command: {e}")
        return JsonResponse({"error": "Failed to execute forecasting command."}, status=500)

    # Fetch any generated plots
    print("[DEBUG] Fetching generated images")
    image_fetch_start = time.time()
    images = get_images_in_directory(settings.BASE_DIR)
    print(f"[DEBUG] Images fetched in {time.time() - image_fetch_start:.2f} seconds")

    # Prepare and return the result
    result = {"content": response}
    if images:
        print(f"[DEBUG] Image path retrieved: {images[0]}")
        result["image_path"] = images[0]  # Return the first image if any

    # Clean up temporary file
    try:
        print("[DEBUG] Removing temporary CSV file")
        os_remove_start = time.time()
        os.remove(csv_file_path)
        print(f"[DEBUG] Temporary file removed in {time.time() - os_remove_start:.2f} seconds")
    except OSError as e:
        print(f"[ERROR] Error removing temporary file: {e}")

    total_time = time.time() - start_time
    print(f"[DEBUG] Exiting handle_forecasting function. Total execution time: {total_time:.2f} seconds")
    return result


import markdown


def markdown_to_html(md_text):
    html_text = markdown.markdown(md_text)
    return html_text


def image_to_base64(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        return None  # Handle the case where the image path is invalid or the image doesn't exist


result = None
response_data = {}

from django.views.decorators.csrf import csrf_exempt


@csrf_exempt
def forecast_sales(request):
    """
    API endpoint to handle sales forecasting requests. It expects a pre-loaded DataFrame.

    Parameters:
    - request: HTTP request object.

    Returns:
    - JSON response with the forecast results.
    """
    print("[DEBUG] Entering forecast_sales function")
    start_time = time.time()

    if request.method == 'POST':
        try:
            print("[DEBUG] Loading DataFrame from 'data1.xlsx'")
            df_load_start = time.time()
            df = pd.read_excel('data1.xlsx')  # Replace with your actual file path
            print(f"[DEBUG] DataFrame loaded successfully in {time.time() - df_load_start:.2f} seconds")

            # Retrieve the user prompt and OpenAI API key
            openai_api_key = os.getenv("OPENAI_API_KEY")
            print(f"[DEBUG] Retrieved OpenAI API key: {openai_api_key}")
            user_prompt = request.POST.get('user_prompt')
            if not user_prompt:
                print("[ERROR] User prompt is missing in the request")
                return JsonResponse({"error": "User prompt is required."}, status=400)

            print(f"[DEBUG] User prompt received: {user_prompt}")

            # Pass the DataFrame to `handle_forecasting`
            print("[DEBUG] Calling handle_forecasting")
            handle_start = time.time()
            result = handle_forecasting(df, openai_api_key, user_prompt, table_name="forecast_table")
            print(f"[DEBUG] handle_forecasting completed in {time.time() - handle_start:.2f} seconds")

            # Prepare the response
            response_data = {}
            if isinstance(result, dict):
                if 'image_path' in result:
                    print(f"[DEBUG] Converting image to Base64: {result['image_path']}")
                    response_data["image_base64"] = image_to_base64(result["image_path"])
                if 'content' in result:
                    print("[DEBUG] Converting AI content to HTML")
                    response_data["content"] = markdown_to_html(result["content"])

            print(f"[DEBUG] Returning successful response. Total time: {time.time() - start_time:.2f} seconds")
            return JsonResponse(response_data, status=200)

        except Exception as e:
            print(f"[ERROR] Exception occurred in forecast_sales: {e}")
            return JsonResponse({"error": "Failed to process forecasting."}, status=500)
    else:
        print(f"[ERROR] Invalid request method. Total time: {time.time() - start_time:.2f} seconds")
        return JsonResponse({"error": "Invalid request method."}, status=400)


# Synthetic Data Generation through wyge
import pandas as pd
from django.views.decorators.csrf import csrf_exempt
from .generator import generate_data_from_text, generate_synthetic_data
import json
import re


def extract_num_rows_from_prompt(user_prompt):
    """
    Extracts the number of rows or records from the user's prompt.
    Supports prompts like:
      - "Generate 100 rows of data"
      - "Generate the 50 records of data"
    """
    match = re.search(r'(\d+)\s+(rows|records)', user_prompt, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def extract_columns_from_prompt(user_prompt):
    """
    Extracts the field names (column names) from the user's prompt.
    Supports prompts like:
      - "field names: S.no, Name, address, First_name"
      - "fields: S.no, Name, address, First_name"
      - "columns: S.no, Name, address, First_name"
      - "field_names: S.no, Name, address, First_name"
      - "column_names: S.no, Name, address, First_name"

    Converts column names to snake_case, removes spaces and special characters.
    Example:
      - "S.no, Name, address, First_name"
      -> ['s_no', 'name', 'address', 'first_name']
    """
    # Look for all possible field identifier formats followed by the column names
    match = re.search(r'(field names|column names|fields|columns|field_names|column_names):\s*([a-zA-Z0-9_,\s\.]+)',
                      user_prompt, re.IGNORECASE)

    if match:
        # Extract the part containing column names
        raw_columns = match.group(2).split(',')
    else:
        return []

    # Format each column name (remove spaces, convert to snake_case, lowercase)
    formatted_columns = [
        re.sub(r'[^a-zA-Z0-9]', '_', col.strip()).lower()
        for col in raw_columns
    ]

    # Remove empty column names and ensure no duplicates
    formatted_columns = list(filter(bool, formatted_columns))
    return list(dict.fromkeys(formatted_columns))  # Remove duplicates


@csrf_exempt
def handle_synthetic_data_api(request):
    """
    API Endpoint to generate synthetic data from a user's prompt.

    Method: POST
    Payload:
      - user_prompt (String): A prompt specifying the number of rows and column names.
      - openai_api_key (String): OpenAI API key
    """
    if request.method == "POST":
        try:
            # Extract user prompt and OpenAI API key from the request
            user_prompt = request.POST.get('user_prompt')
            openai_api_key = os.getenv('OPENAI_API_KEY')

            # Validate user prompt
            if not user_prompt or not openai_api_key:
                return JsonResponse({"error": "Missing required parameters: user_prompt or OpenAI API key"}, status=400)

            # Extract number of rows from the prompt
            num_rows = extract_num_rows_from_prompt(user_prompt)
            if num_rows is None:
                return JsonResponse({"error": "Number of rows or records not found in the prompt."}, status=400)

            # Extract column names from the prompt
            column_names = extract_columns_from_prompt(user_prompt)
            print(column_names)
            if not column_names:
                return JsonResponse({"error": "No field names found in the prompt."}, status=400)

            # Generate synthetic data
            generated_df = generate_data_from_text(openai_api_key, user_prompt, column_names, num_rows=num_rows)

            # Convert to CSV format
            combined_csv = generated_df.to_csv(index=False)

            return JsonResponse({

                "data": combined_csv
            }, status=200)

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request method. Use POST."}, status=405)


# For extended_synthetic_data
import tempfile
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import os


@csrf_exempt
def handle_synthetic_data_extended(request):
    """
    API Endpoint to generate synthetic data from a user's uploaded file and prompt.

    Method: POST
    Payload:
      - uploaded_file (File): The empty Excel or CSV file with column names
      - user_prompt (String): A prompt specifying the number of rows
      - openai_api_key (String): OpenAI API key
    """
    print("[DEBUG] Entering handle_synthetic_data_extended function")

    if request.method == "POST":
        try:
            print("[DEBUG] Handling POST request")

            # Extract uploaded file, user prompt, and API key from the request
            uploaded_file = request.FILES.get('file')
            user_prompt = request.POST.get('user_prompt')
            openai_api_key = os.getenv('OPENAI_API_KEY')

            print(f"[DEBUG] Uploaded file: {uploaded_file}")
            print(f"[DEBUG] User prompt: {user_prompt}")
            print(f"[DEBUG] OpenAI API key: {'Provided' if openai_api_key else 'Missing'}")

            if not uploaded_file or not user_prompt or not openai_api_key:
                print("[ERROR] Missing required parameters")
                return JsonResponse({"error": "Missing required parameters"}, status=400)

            # Determine file type and extract column names
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            print(f"[DEBUG] File extension: {file_extension}")

            if file_extension == ".xlsx":
                print("[DEBUG] Reading Excel file")
                df = pd.read_excel(uploaded_file)
            elif file_extension == ".csv":
                print("[DEBUG] Reading CSV file")
                df = pd.read_csv(uploaded_file)
            else:
                print("[ERROR] Unsupported file format")
                return JsonResponse({"error": "Unsupported file format. Please upload an Excel or CSV file."},
                                    status=400)

            print(f"[DEBUG] Initial DataFrame columns: {list(df.columns)}")

            # Create a temporary file for the data
            with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as temp_file:
                temp_file_name = temp_file.name
                print(f"[DEBUG] Temporary file created at: {temp_file_name}")

                # Save the truncated or original data to a temporary location
                if file_extension == ".xlsx":
                    print("[DEBUG] Saving DataFrame to temporary Excel file")
                    df.to_excel(temp_file_name, index=False)
                elif file_extension == ".csv":
                    print("[DEBUG] Saving DataFrame to temporary CSV file")
                    df.to_csv(temp_file_name, index=False)

            # Extract the number of rows from the prompt
            print("[DEBUG] Extracting number of rows from the user prompt")
            num_rows = extract_num_rows_from_prompt(user_prompt)
            print(f"[DEBUG] Number of rows extracted: {num_rows}")

            if num_rows is None:
                print("[ERROR] Number of rows not found in the prompt")
                return JsonResponse({"error": "Number of rows not found in the prompt"}, status=400)

            # Generate synthetic data using the temporary file path
            print(f"[DEBUG] Generating synthetic data with {num_rows} rows")
            generated_df = generate_synthetic_data(openai_api_key, temp_file_name, num_rows)
            print(f"[DEBUG] Synthetic data generated successfully: {generated_df.shape[0]} rows")

            # Combine the original and synthetic data
            print("[DEBUG] Combining original and synthetic data")
            combined_df = pd.concat([df, generated_df], ignore_index=True)
            print(f"[DEBUG] Combined DataFrame shape: {combined_df.shape}")

            # Convert to CSV for download
            print("[DEBUG] Converting combined DataFrame to CSV format")
            combined_csv = combined_df.to_csv(index=False)

            print("[DEBUG] Returning successful response")
            return JsonResponse({
                "data": combined_csv
            }, status=200)

        except Exception as e:
            print(f"[ERROR] Exception occurred: {e}")
            return JsonResponse({"error": str(e)}, status=500)

    print("[ERROR] Invalid request method")
    return JsonResponse({"error": "Invalid request method. Use POST."}, status=405)


# SAP SYSTEM
from .database import HanaDBManager

db1 = HanaDBManager()


@csrf_exempt
def hana_connection(request):
    global connection_obj
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        host = request.POST['host']
        port = request.POST['port']
        print("all details received......")
        connection_obj = db1.connect_to_hana(username, password, host, port)
        print(connection_obj)
        return HttpResponse(json.dumps({"tables": connection_obj}), content_type="application/json")


@csrf_exempt
def upload_data(request):
    if request.method == 'POST':
        # Retrieve the email and uploaded file
        email = request.POST.get('mail')
        files = request.FILES.get('file')  # Retrieve the uploaded file

        if not files:
            return HttpResponse('No files uploaded', status=400)

        file_name = files.name
        file_extension = os.path.splitext(file_name)[1].lower()  # Extract file extension

        try:
            # Process the uploaded file based on its extension
            if file_extension == '.csv':
                # Read CSV file
                content = files.read().decode('utf-8')
                csv_data = io.StringIO(content)
                df = pd.read_csv(csv_data)
            elif file_extension in ['.xls', '.xlsx']:
                # Read Excel file
                df = pd.read_excel(files)
            else:
                # Unsupported file type
                raise SuspiciousOperation("Unsupported file format")

            # Save the uploaded file locally for backup/logging purposes
            upload_dir = "uploads2"
            os.makedirs(upload_dir, exist_ok=True)

            # For CSV file
            csv_file_path = os.path.join(upload_dir, file_name.replace(file_extension, '.csv').lower())
            df.to_csv(csv_file_path, index=False)

            # Save as Excel file
            excel_file_path = os.path.join(upload_dir, file_name.replace(file_extension, '.xlsx').lower())
            df.to_excel(excel_file_path, index=False, engine='openpyxl')  # Use openpyxl as the Excel writer

            # Save a working copy as 'data.csv'
            df.to_csv('data2.csv', index=False)
            df.to_excel('data2.xlsx', index=False, engine='openpyxl')  # Excel

            # Insert the data into the HANA database
            results = db1.save_or_update_record(email, df.to_dict(orient='records'), file_name)

            # # Perform data analysis on the DataFrame
            # response_data1 = analyze_data(df)  # Assuming `analyze_data` is a function that analyzes data
            #
            # # Return the analytics response along with the first 10 rows
            # response_data1['preview'] = df.head(10).to_dict(orient='records')
            # response_data1['upload_status'] = results  # Include database insert result

            return JsonResponse("Records inserted successfully", safe=False)

        except Exception as e:
            # Handle errors during file processing or database interaction
            print(e)
            return HttpResponse(f"Failed to upload and analyze file: {str(e)}", status=500)

    # If the request method is not POST
    return HttpResponse("Invalid Request Method", status=405)


@csrf_exempt
def reading_data(request):
    if request.method == 'POST':
        try:
            # Get the table name from the request
            tablename = request.POST['tablename']
            print(tablename)

            # Fetch data from the specified table
            df = db1.get_tables_data(tablename)

            if df.empty:
                return HttpResponse("No data found or table does not exist.", status=404)

            # Save data to CSV in the current directory
            csv_filename = os.path.join("uploads2", f"{tablename.lower()}.csv")
            df.to_csv(csv_filename, index=False)

            # Optionally, you can also save it in a different location or give a custom name
            df.to_csv('data2.csv', index=False)

            # Return the data as JSON in the response
            return HttpResponse(df.to_json(), content_type="application/json")

        except Exception as e:
            return HttpResponse(f"Error: {str(e)}", status=500)


@csrf_exempt
def flespicred(request):
    if request.method == 'POST':
        return HttpResponse("Success")


@csrf_exempt
def download_flespi_data(request):
    # https://flespi.io/gw/devices/5439260/messages
    # axLBthbazeJkKKkpr2sVK9rAeXfFJGmH1V9k18iqaSyKqHYHzetadIyitBL15WyU
    if request.method == 'POST':
        flespi_URL = request.POST.get('flespi_URL')
        flespi_token = request.POST.get('flespi_token')
        try:
            current_datetime = datetime.now(tz=ZoneInfo('Asia/Kolkata'))
            start_of_day = current_datetime.replace(month=current_datetime.month - 1, day=current_datetime.day,
                                                    hour=current_datetime.hour, minute=current_datetime.minute,
                                                    second=0,
                                                    microsecond=0)
            response = requests.get(
                f'{flespi_URL}?data=%7B%22from%22%3A{start_of_day.timestamp()}%2C%22to%22%3A{datetime.now().timestamp()}%7D',
                headers={
                    'Authorization':
                        f'FlespiToken {flespi_token}'
                })
            multi_data = json.loads(response.text)['result']
            multi_data = pre_process_multi_data(multi_data)
            return HttpResponse(json.dumps({"data": multi_data}), content_type="application/json")
        except Exception as e:
            return HttpResponse(str(e))


def pre_process_multi_data(multi_data):
    for idx, record in enumerate(multi_data):
        multi_data[idx].update({
            "timestamp": datetime.fromtimestamp(multi_data[idx]["timestamp"],
                                                tz=ZoneInfo('Asia/Kolkata')).strftime("%Y-%m-%d %H-%M-%S"),
            "server.timestamp": datetime.fromtimestamp(multi_data[idx]["server.timestamp"],
                                                       tz=ZoneInfo('Asia/Kolkata')).strftime(
                "%Y-%m-%d %H-%M-%S")
        })
    return multi_data


# KPI APIS
from collections import defaultdict

KPI_LOGICS = defaultdict()
checks = []


##KPI process code
@csrf_exempt
def get_prompt(request):
    """
    Handles POST requests to process a user prompt, analyze a CSV file, generate KPIs,
    and store them in a JSON file.
    """
    try:
        if request.method != "POST":
            return HttpResponse("Invalid request method. Only POST requests are allowed.", status=405)

        # Initialize variables
        global KPI_LOGICS, checks
        KPI_LOGICS = defaultdict()
        checks = []

        # Extract the prompt from POST request
        prompt = request.POST.get('prompt')
        if not prompt:
            return HttpResponse("Prompt is required.", status=400)

        # Check if required file exists
        processed_data_path = os.path.join('uploads', 'processed_data.csv')
        if not os.path.exists(processed_data_path):
            return HttpResponse("No processed data file found.", status=404)

        # Read and save data.csv
        df = pd.read_csv(processed_data_path)
        df.to_csv('data.csv', index=False)

        # Prepare the prompt description for the analytics bot
        prompt_desc = (
            f"You are analytics_bot. Analyse the data: {df.head()} and for the user query '{prompt}', "
            f"generate KPIs with response as KPI Name, Column, and Logic. Response should be in Python dictionary format "
            f"with KPI names as keys. In response, don't add any other information, just provide the response dictionary."
        )

        n = 2  # Retry logic for generating KPIs
        kpis = {}
        while n > 0:
            res, kpis = generate_code_kpi(prompt_desc)  # Assuming generate_code is a valid function
            if res is not None:
                # Load existing KPIs from kpis.json if it exists, otherwise create an empty dictionary
                kpis_store_path = 'kpis.json'
                kpis_store = {}
                if os.path.exists(kpis_store_path):
                    with open(kpis_store_path, 'r') as fp:
                        kpis_store = json.load(fp)

                # Update kpis.json with new KPIs
                with open(kpis_store_path, 'w') as fp:
                    kpis_store.update(kpis)
                    json.dump(kpis_store, fp)
                break  # Exit loop if generation was successful
            n -= 1  # Decrement retry count

        # Check if KPI configuration exists and load additional KPIs from it
        kpi_config_path = os.path.join('uploads', 'kpi_config.json')
        if os.path.exists(kpi_config_path):
            with open(kpi_config_path, 'r') as json_file:
                kpis_dict = json.load(json_file)
            for kpi in kpis_dict.get('Kpis', {}).get('kpi', []):
                kpi_name = kpi.get('KPI_Name')
                if kpi_name:
                    kpis[kpi_name] = kpi
                    checks.append(kpi_name)

        # Return a JSON response with KPI data and checks
        return JsonResponse({
            'status': 'success',
            'kpis': kpis,
            'checks': checks
        })

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        print(error_message)  # Log the error
        return HttpResponse(error_message, status=500)


@csrf_exempt
def generate_code_kpi(prompt_eng):
    try:
        global KPI_LOGICS
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt_eng}
            ]
        )
        all_text = ""
        # Display generated content dynamically
        for choice in response.choices:
            message = choice.message
            chunk_message = message.content if message else ''
            all_text += chunk_message
        all_text = all_text.lower().replace('```python', '').replace('```', '')
        print(all_text)
        data_dict = json.loads(all_text)
        print("datadict", data_dict)
        for key, value in data_dict.items():
            if 'kpi name' in value:
                kpi_name = value['kpi name']
            elif 'name' in value:
                kpi_name = value["name"]
            else:
                kpi_name = key
            KPI_LOGICS[key] = {
                "KPI Name": kpi_name,
                "Column": value["column"],
                "Logic": value["logic"]
            }
        return all_text, KPI_LOGICS
    except Exception as e:
        print(e)
        return None, None


# For getting the KPI codes
def generate_kpi_code(kpi_list):
    """
    Generates Python code for a list of KPIs, saves plots, and returns file paths,
    a list of Base64-encoded images, and the generated code.
    """
    try:
        # Load and process data
        df = pd.read_csv("data.csv")
        df = updatedtypes(df)

        codes = ''
        paths = {}
        base64_images = []  # List to store only Base64-encoded images

        charts_dir = os.path.join(os.getcwd(), 'static', 'charts')

        if not os.path.exists(charts_dir):
            os.makedirs(charts_dir)

        # Clear existing charts
        for f in os.listdir(charts_dir):
            file_path = os.path.join(charts_dir, f)
            if os.path.isfile(file_path):
                os.remove(file_path)

        for kpi in kpi_list:
            prompt_desc = (
                f"You are analytics_bot. Read the data from data.csv file with example data as {df.head()} and generate python code with kpi details as {KPI_LOGICS.get(kpi, {})}. "
                f"Save result in variable named result, plot a suitable plot for the result obtained, save it as name based on kpi and use static/charts to save the file. "
                f"If length of result variable is 1 then keep bar width thin and x-axis limit as -0.5 and 0.5."
            )

            code = ''
            try:
                code += generate_code2(prompt_desc)
            except Exception as e:
                print(f"Code generation failed for {kpi}: {str(e)}")
                code += f'Code generation failed for {kpi}'

            codes += f"<b>{kpi.capitalize()}</b>\n{code}\n"

        if os.path.exists(charts_dir):
            for path in os.listdir(charts_dir):
                if path.endswith(('.png', '.jpg', '.jpeg')):  # Only include image files
                    image_path = os.path.join(charts_dir, path)

                    # Read the image and convert to Base64
                    with open(image_path, 'rb') as image_file:
                        base64_encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
                        base64_images.append(base64_encoded_image)  # Append only Base64 string

        return paths, base64_images, codes
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        print(error_message)
        return {}, [], error_message


@csrf_exempt
def kpi_code(request):
    try:
        if request.method != "POST":
            return JsonResponse({"error": "Invalid request method. Only POST requests are allowed."}, status=405)

        # Extract KPI names from POST request
        kpi_list = request.POST.getlist("kpi_names")
        if not kpi_list:
            return JsonResponse({"error": "KPI names are required."}, status=400)

        # Generate paths, base64 images, and code for the provided KPIs
        paths, base64_images, codes = generate_kpi_code(kpi_list)  # Assuming generate_kpi_code is a valid function

        # Return paths, base64 images, and code as JSON response
        return JsonResponse({
            'status': 'success',
            # 'paths': paths,
            'base64_images': base64_images,
            'code': codes,
            'kpis': KPI_LOGICS,  # Assuming KPI_LOGICS is a global or properly imported variable
            'checks': checks  # Assuming checks is a global or properly imported variable
        })
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        print(error_message)  # Log the error
        return JsonResponse({"error": error_message}, status=500)


@csrf_exempt
def generate_code2(prompt_eng):
    trials = 2
    try:
        while trials > 0:
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt_eng}
                ]
            )
            all_text = ""

            # Display generated content dynamically
            for choice in response.choices:
                print(f"Debug - choice structure: {choice}")  # Debugging line
                message = choice.message
                print(f"Debug - message structure: {message}")  # Debugging line
                chunk_message = message.content if message else ''
                all_text += chunk_message

            print(all_text)
            python_chunks = all_text.count("```python")
            idx = 0
            code = ''
            for i in range(python_chunks):
                code_start = all_text[idx:].find("```python") + 9
                code_end = all_text[idx:].find("```", code_start)
                code += all_text[idx:][code_start:code_end]
                idx = code_end
            print(code)
            try:
                local_vars = {}
                exec(code, {}, local_vars)
                code += f"\n <b>Output: {local_vars['result']}</b> \n <hr>"
                return code
            except Exception as e:
                print(e)
                trials -= 1
    except Exception as e:
        print(e)


# Code for Predefined KPIS
# step1:Detecting the type from the dataset,whether the data is sustainability data/oem/safety type

def analyze_dataset_with_llm(df):
    """
    Simulate LLM logic to analyze the dataset and detect the type (Sustainability, Safety, or OEM),
    along with the reasoning behind the classification.
    """
    prompt = f"""
    You are an AI expert system that classifies datasets into one of three types: 
    - **Sustainability**: Measures environmental and social impact, e.g., carbon emissions, energy efficiency, water usage, and waste recycling.
    - **Safety**: Tracks workplace safety performance, e.g., incident rates, near-miss reports, safety training, and days without accidents.
    - **OEM (Original Equipment Manufacturer)**: Evaluates manufacturing efficiency, e.g., production output, machine uptime, defect rates, and on-time delivery.

    Here are the first 5 rows of the dataset and the column names:
    Columns: {list(df.columns)}
    Sample Data: {df.head().to_dict()}

    Based on this information, determine the most likely type (Sustainability, Safety, or OEM) that best describes this dataset. 
    Additionally, explain why you have classified it under this type, citing specific columns or data features that influenced the decision.
    Please provide your response in the following format:
    1. Type: [Classification Type]
    2. Explanation: [Reasoning for classification]

    Only respond with the type and the explanation, and do not provide any additional information.
    """

    try:
        # Call the OpenAI API
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )

        all_text = ""

        # Process the response and extract the classification and reasoning
        for choice in response.choices:
            message = choice.message
            chunk_message = message.content if message else ''
            all_text += chunk_message

        print(f"LLM Response: {all_text}")  # Print the LLM response for debugging
        lines = all_text.splitlines()
        classification_type = lines[0].split(":")[1].strip() if lines else None
        return classification_type

    except Exception as e:
        print(f"Error calling OpenAI API: {str(e)}")
        return None  # Return None if there was an error


# actual api for getting the response as type
@csrf_exempt
def getting_types(request):
    """
    Handles POST requests to process a user prompt, analyze a CSV file from a fixed folder,
    and detect the type (Sustainability, Safety, or OEM) using an LLM.
    """
    try:
        if request.method != "POST":
            return HttpResponse("Invalid request method. Only POST requests are allowed.", status=405)

        # Check if required file exists
        processed_data_path = os.path.join('uploads', 'processed_data.csv')
        if not os.path.exists(processed_data_path):
            return HttpResponse("No processed data file found.", status=404)

        # Read and save data.csv
        try:
            df = pd.read_csv(processed_data_path)
        except Exception as e:
            return HttpResponse(f"Error reading the CSV file: {str(e)}", status=500)

        print("Dataset Columns:")
        print(df.head(5))  # Print the first 5 rows for debugging
        df.to_csv('data.csv', index=False)  # Save a copy of the file (optional)

        # Call the LLM to detect the type
        detected_type = analyze_dataset_with_llm(df)

        # If no type was detected, return a message indicating no match
        if not detected_type or detected_type not in ['Sustainability', 'Safety', 'OEM']:
            return JsonResponse({
                'status': 'failure',
                'message': 'No relevant type detected for Sustainability, Safety, or OEM.'
            })

        # Return the detected type
        return JsonResponse({
            'status': 'success',
            'type': detected_type  # Return only the detected type
        })

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        print(error_message)  # Log the error
        return HttpResponse(error_message, status=500)


#Getting prepared the predefined kpis with this
@csrf_exempt
def predefined_kpi_getting(request):
    """
    Handles POST requests to process a user query, analyze a CSV file, generate KPIs based on type and category,
    and store them in a JSON file.
    """
    try:
        if request.method != "POST":
            return HttpResponse("Invalid request method. Only POST requests are allowed.", status=405)

        # Initialize variables
        global KPI_LOGICS, checks
        KPI_LOGICS = defaultdict()
        checks = []

        # Extract the type and category from the POST request
        kpi_type = request.POST.get('type')
        kpi_category = request.POST.get('category')

        if not kpi_type or not kpi_category:
            return HttpResponse("Both 'type' and 'category' are required.", status=400)

        # Check if required file exists
        processed_data_path = os.path.join('uploads', 'processed_data.csv')
        if not os.path.exists(processed_data_path):
            return HttpResponse("No processed data file found.", status=404)

        # Read and save data.csv
        df = pd.read_csv(processed_data_path)
        df.to_csv('data.csv', index=False)

        # Prepare the prompt description for the analytics bot using the type and category
        prompt_desc = (
            f"You are analytics_bot. Analyse the data: {df.head()} and based on the type '{kpi_type}' "
            f"and category '{kpi_category}', generate KPIs with the response as KPI Name, Column, and Logic. "
            f"Response should be in Python dictionary format with KPI names as keys. "
            f"In the response, don't add any other information, just provide the response dictionary."
        )

        n = 2  # Retry logic for generating KPIs
        kpis = {}
        while n > 0:
            res, kpis = generate_code_kpi(prompt_desc)  # Assuming generate_code_kpi is a valid function
            if res is not None:
                # Load existing KPIs from kpis.json if it exists, otherwise create an empty dictionary
                kpis_store_path = 'kpis.json'
                kpis_store = {}
                if os.path.exists(kpis_store_path):
                    with open(kpis_store_path, 'r') as fp:
                        kpis_store = json.load(fp)

                # Update kpis.json with new KPIs
                with open(kpis_store_path, 'w') as fp:
                    kpis_store.update(kpis)
                    json.dump(kpis_store, fp)
                break  # Exit loop if generation was successful
            n -= 1  # Decrement retry count

        # Check if KPI configuration exists and load additional KPIs from it
        kpi_config_path = os.path.join('uploads', 'kpi_config.json')
        if os.path.exists(kpi_config_path):
            with open(kpi_config_path, 'r') as json_file:
                kpis_dict = json.load(json_file)
            for kpi in kpis_dict.get('Kpis', {}).get('kpi', []):
                kpi_name = kpi.get('KPI_Name')
                if kpi_name:
                    kpis[kpi_name] = kpi
                    checks.append(kpi_name)

        # Return a JSON response with KPI data and checks
        return JsonResponse({
            'status': 'success',
            'kpis': kpis,
            'checks': checks
        })

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        print(error_message)  # Log the error
        return HttpResponse(error_message, status=500)


# For Model checking....
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import pmdarima as pm
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from datetime import timedelta, datetime
from dateutil.relativedelta import relativedelta
import xml.etree.ElementTree as ET
from keras.models import load_model
import matplotlib.pyplot as plt


@csrf_exempt
def models(request):
    try:
        # Check if the processed data file exists
        processed_data_path = os.path.join("uploads", 'processed_data.csv')
        if not os.path.exists(processed_data_path):
            return JsonResponse({"msg": "Please upload file to continue."}, status=404)

        # Read CSV file
        df = pd.read_csv(processed_data_path)
        rf_result = request.session.get('rf_result', '')
        if rf_result:
            request.session['rf_result'] = ''
            return JsonResponse({
                'form1': True,
                'columns': list(df.columns),
                'rf_result': rf_result
            })

        if request.method == 'POST':
            model_type = request.POST.get('model')
            col = request.POST.get('col')
            request.session['col_predict'] = col

            if model_type == 'RandomForest':
                stat, cols = random_forest(df, col)  # Assuming random_forest is a valid function
                return JsonResponse({
                    'form1': True,
                    'columns': list(df.columns),
                    'rf': True,
                    'status': stat,
                    'rf_cols': cols
                })

            elif model_type == 'K-Means':
                stat, clustered_data = kmeans_train(df)  # Assuming kmeans_train is a valid function
                return JsonResponse({
                    'form1': True,
                    'columns': list(df.columns),
                    'cluster': True,
                    'status': stat,
                    'clustered_data': clustered_data
                })

            elif model_type == 'Arima':
                stat = arima_train(df, col)  # Assuming arima_train is a valid function
                path = f"../models/arima/{col}/actual_vs_forecast.png"
                return JsonResponse({
                    'form1': True,
                    'columns': list(df.columns),
                    'status': stat,
                    'arima': True,
                    'path': path
                })

            elif model_type == 'OutlierDetection':
                res = detect_outliers_zscore(df, col)  # Assuming detect_outliers_zscore is a valid function
                return JsonResponse({
                    'form1': True,
                    'columns': list(df.columns),
                    'status': True,
                    'processed_data': res,
                    'OutlierDetection': True
                })

        # Default GET response
        return JsonResponse({
            'form1': True,
            'columns': list(df.columns)
        })

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        print(error_message)  # Log the error
        return JsonResponse({
            'form1': False,
            'msg': error_message
        }, status=500)


def outliercheck(df, column):
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f'detect outliers for  the following data {df[column]}'}
        ]
    )
    all_text = ""
    # Display generated content dynamically
    for choice in response.choices:
        message = choice.message
        chunk_message = message.content if message else ''
        all_text += chunk_message
    print(all_text)
    return all_text


def detect_outliers_zscore(df, column, threshold=3):
    try:

        res = outliercheck(df, column)

        # # Select numeric columns only
        # numeric_cols = df.select_dtypes(include=np.number)
        #
        # # Calculate Z-Scores for each numeric column
        # z_scores = (numeric_cols - numeric_cols.mean()) / numeric_cols.std()
        #
        # # Calculate an aggregate Z-Score for each row (e.g., max absolute Z-Score)
        # df['Row_Z-Score'] = z_scores.abs().max(axis=1)
        #
        # # Flag rows where the aggregate Z-Score exceeds the threshold
        # df['Outlier'] = df['Row_Z-Score'].apply(lambda x: 'Yes' if x > threshold else 'No')
        # df.drop('Row_Z-Score', axis=1,inplace=True)
        return res
    except Exception as e:
        print(e)


def find_elbow_point(inertia_values):
    # Calculate the rate of change between successive inertia values
    changes = np.diff(inertia_values)
    # Identify the elbow as the point where change starts to decrease
    elbow_point = np.argmin(np.abs(np.diff(changes))) + 1
    return elbow_point


def arima_train(data, target_col):
    try:
        # Identify date column by checking for datetime type
        date_column = None
        if not os.path.exists(os.path.join("models", 'arima', target_col)):
            os.makedirs(os.path.join("models", 'arima', target_col), exist_ok=True)
            for col in data.columns:
                if data.dtypes[col] == 'object':
                    try:
                        # Attempt to convert column to datetime
                        pd.to_datetime(data[col])
                        date_column = col
                        break
                    except (ValueError, TypeError):
                        continue
            if not date_column:
                raise ValueError("No datetime column found in the dataset.")
            print(date_column)
            # Set the date column as index
            data[date_column] = pd.to_datetime(data[date_column])
            data.set_index(date_column, inplace=True)
            print(data.head(15))
            # Identify forecast columns (numeric columns)
            forecast_columns = data.select_dtypes(include=[np.number]).columns.tolist()
            if not forecast_columns:
                raise ValueError("No numeric columns found for forecasting in the dataset.")

            # Infer frequency of datetime index
            freq = pd.infer_freq(data.index)
            print(date_column, freq)
            # Determine m based on inferred frequency
            # Determine m based on inferred frequency
            if freq == '15T':  # Quarter-hourly data (every 15 minutes)
                m = 96  # Daily seasonality (96 intervals in a day)
            elif freq == '30T':  # Half-hourly data (every 30 minutes)
                m = 48  # Daily seasonality (48 intervals in a day)
            elif freq == 'H':  # Hourly data
                m = 24  # Daily seasonality (24 intervals in a day)
            elif freq == 'D':  # Daily data
                m = 7  # Weekly seasonality (7 days in a week)
            elif freq == 'W':  # Weekly data
                m = 52  # Yearly seasonality (52 weeks in a year)
            elif freq == 'M':  # Monthly data
                m = 12  # Yearly seasonality (12 months in a year)
            elif freq == 'Q':  # Quarterly data
                m = 4  # Yearly seasonality (4 quarters in a year)
            elif freq == 'A' or (freq and freq.startswith('A-')):  # Annual data (any month-end)
                m = 1  # No further seasonality within a year
            else:
                raise ValueError(f"Unsupported frequency '{freq}'. Ensure data is in a common time interval.")
            results = {}
            try:
                data_actual = data[target_col].dropna()  # Remove NaNs if any

                # Split data into train and test sets
                train = data_actual.iloc[:-m]
                test = data_actual.iloc[-m:]

                # Auto ARIMA model selection
                model = pm.auto_arima(train,
                                      m=m,  # frequency of seasonality
                                      seasonal=True,  # Enable seasonal ARIMA
                                      d=None,  # determine differencing
                                      test='adf',  # adf test for differencing
                                      start_p=0, start_q=0,
                                      max_p=12, max_q=12,
                                      D=None,  # let model determine seasonal differencing
                                      trace=True,
                                      error_action='ignore',
                                      suppress_warnings=True,
                                      stepwise=True)
                # Forecast and calculate errors
                fc, confint = model.predict(n_periods=m, return_conf_int=True)
                # Save results to dictionary
                results = {
                    "actual": {
                        "date": list(test.index.astype(str)),
                        "values": [float(val) if isinstance(val, np.float_) else int(val) for val in
                                   test.values]
                    },
                    "forecast": {
                        "date": list(test.index.astype(str)),
                        "values": [float(val) if isinstance(val, np.float_) else int(val) for val in fc]
                    }
                }
                if not os.path.exists(os.path.join("models", 'arima', target_col)):
                    os.makedirs(os.path.join("models", 'arima', target_col), exist_ok=True)
                with open(os.path.join("models", 'arima', target_col, target_col + '_results.json'), 'w') as fp:
                    json.dump(results, fp)
                    plot_graph(results, os.path.join('models', 'arima', target_col))
                print(f"Results saved to {os.path.join('models', 'arima', target_col, target_col + '_results.json')}")
            except Exception as e:
                print(e)
                return False
            return True
    except Exception as e:
        print(e)
        return False


def plot_graph(data, file_path):
    try:
        col = file_path.split('\\')[-1]
        actual_dates = [datetime.strptime(date, "%Y-%m-%d") for date in data["actual"]["date"]]
        forecast_dates = [datetime.strptime(date, "%Y-%m-%d") for date in data["forecast"]["date"]]

        # Extract values
        actual_values = data["actual"]["values"]
        forecast_values = data["forecast"]["values"]

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(actual_dates, actual_values, label='Actual', color='blue', marker='o')
        plt.plot(forecast_dates, forecast_values, label='Forecast', color='orange', linestyle='--', marker='x')

        # Formatting
        plt.title(f'{col} Actual vs Forecast Values Over Time')
        plt.xlabel('Date')
        plt.ylabel('Values')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()

        # Save the plot as a PNG image
        plt.savefig(os.path.join(file_path, "actual_vs_forecast.png"), format="png", dpi=300)
    except Exception as e:
        print(e)


def kmeans_train(data):
    try:
        # Identify categorical and numerical columns
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_columns = data.select_dtypes(include=[np.number]).columns.tolist()

        # Handle missing values (if any)
        imputer = SimpleImputer(strategy='mean')
        data[numerical_columns] = imputer.fit_transform(data[numerical_columns])
        joblib.dump(imputer, 'imputer.pkl')

        # Build a transformer for preprocessing: scaling numerical columns and encoding categorical columns
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_columns),  # Standard scaling for numerical columns
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
                # One-Hot encoding for categorical columns
            ])

        # Apply preprocessing and fit KMeans
        X = preprocessor.fit_transform(data)

        # Find the optimal k using the elbow method with KMeans
        inertia = []
        K_range = range(1, 11)
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=0)
            kmeans.fit(X)
            inertia.append(kmeans.inertia_)

        # Determine the optimal k
        optimal_k = find_elbow_point(inertia)
        print('Optimal number of clusters (k) based on the Elbow Method:', optimal_k)

        # Initialize KMeans with the optimal number of clusters
        kmeans = KMeans(n_clusters=optimal_k, random_state=0)

        # Fit KMeans to the preprocessed data
        kmeans.fit(X)

        # Save the trained model and preprocessor
        joblib.dump(kmeans, 'kmeans_model.pkl')  # Save KMeans model
        joblib.dump(preprocessor, 'preprocessor.pkl')  # Save Preprocessing pipeline

        # Add cluster labels to the original data
        data['Cluster'] = kmeans.labels_
        return True, data
    except Exception as e:
        print(e)
        return False, data


def load_pipeline(save_path="model_pipeline.pkl"):
    # Load the saved pipeline
    pipeline = joblib.load(save_path)
    print(f"Pipeline loaded from: {save_path}")
    return pipeline


def random_forest(data, target_column):
    try:
        if not os.path.exists(os.path.join("models", "rf", target_column, 'deployment.json')):
            os.makedirs(os.path.join("models", "rf", target_column), exist_ok=True)
            # Separate features and target
            X = data.drop(columns=[target_column])
            y = data[target_column]

            # Detect categorical and numerical features
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns
            numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

            # Preprocessing pipelines for numerical and categorical data
            numerical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())])

            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))])

            # Combine preprocessing steps
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, numerical_cols),
                    ('cat', categorical_transformer, categorical_cols)
                ])

            # Choose Random Forest type based on target type
            if y.nunique() <= 5:  # Classification for few unique target values
                model_type = 'Classification'
                model = RandomForestClassifier(random_state=42)
            else:  # Regression for continuous target values
                model_type = 'Regression'
                model = RandomForestRegressor(random_state=42)

            # Create pipeline
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', model)
            ])

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train the pipeline
            pipeline.fit(X_train, y_train)

            cv = min(5, len(X_test))

            # Evaluate the model using cross-validation
            scores = cross_val_score(pipeline, X_test, y_test, cv=cv)
            print(f"Model Performance (CV): {scores.mean():.4f} ± {scores.std():.4f}")

            # Save the pipeline
            joblib.dump(pipeline, os.path.join("models", "rf", target_column, "pipeline.pkl"))
            print(f'Pipeline saved to: {os.path.join("models", "rf", target_column, "pipeline.pkl")}')

            with open(os.path.join("models", "rf", target_column, "deployment.json"), "w") as fp:
                json.dump({"columns": list(X_train.columns), "model_type": model_type, "Target_column": target_column},
                          fp, indent=4)
            return True, list(X_train.columns)
        else:
            with open(os.path.join(os.getcwd(), "models", "rf", target_column, 'deployment.json'), "r") as fp:
                data = json.load(fp)
            return True, data['columns']
    except Exception as e:
        print(e)
        return False, []


# Model prediction for random forest
from django.shortcuts import redirect


@csrf_exempt
def model_predict(request):
    try:
        if request.POST.get('form_name') == 'rf':
            res = {}
            for col in request.POST:
                res.update({col: request.POST[col]})
            del res['form_name']
            df = pd.DataFrame([res])
            loaded_pipeline = load_pipeline(
                os.path.join("models", "rf", request.session['col_predict'], "pipeline.pkl"))
            predictions = loaded_pipeline.predict(df)
            print(predictions)
            request.session['rf_result'] = predictions[0]
            return redirect('models')
    except Exception as e:
        print(e)
