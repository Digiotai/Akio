from asyncio.log import logger
import base64
import io
import json
import os
import sys
from io import StringIO
import pandas as pd

import requests
from django.conf import settings
from django.contrib.auth import authenticate, login, logout
from django.core.exceptions import SuspiciousOperation
from django.http import HttpResponse, HttpResponseBadRequest, JsonResponse
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from dotenv import load_dotenv
from openai import OpenAI
from digiotai.digiotai_jazz import Agent, Task, OpenAIModel, SequentialFlow, InputType, OutputType
#from .database import PostgreSQLDB
from .form import CreateUserForm
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse
import json
import base64
import pandas as pd
import matplotlib.pyplot as plt

import json

from django.shortcuts import render
import os
import pandas as pd
from django.core.files.storage import default_storage
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from .database import *
import io
import hashlib
import joblib
import numpy as np
from keras.models import load_model

global connection_obj
db = PostgresDatabase()

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

#db = PostgreSQLDB(dbname='uibmogli', user='uibmogli', password='8ogImHfL_1G249lXtM3k2EAIWTRDH2mX')


def get_csv_metadata(df):
    metadata = {
        "columns": df.columns.tolist(),
        "data_types": df.dtypes.to_dict(),
        "null_values": df.isnull().sum().to_dict(),
        "example_data": df.head().to_dict()
    }
    return metadata
#
# #login page
# @csrf_exempt
# def loginpage(request):
#     if request.method == 'POST':
#         username = request.POST.get('username')
#         password = request.POST.get('password')
#         user = authenticate(request, username=username, password=password)
#         if user is not None:
#             login(request, user)
#             user_details = db.get_user_data(username)
#             return HttpResponse(json.dumps({"status": "Success", "user_details": user_details}),
#                                 content_type="application/json")
#         else:
#             return HttpResponse('User Name or Password is incorrect')
#     return HttpResponse("Login failed")
#
#
# #logout page
# @csrf_exempt
# def logoutpage(request):
#     try:
#         logout(request)
#         request.session.clear()
#         return redirect('demo:login')
#     except Exception as e:
#         return HttpResponse(str(e))
#
#
# #register page
# @csrf_exempt
# def register(request):
#     if request.method == 'POST':
#         form = CreateUserForm(request.POST)
#         try:
#             if form.is_valid():
#                 form.save()
#                 user = form.cleaned_data.get('username')
#                 email = form.cleaned_data.get('email')
#                 db.add_user(user, email)
#                 user_details = db.get_user_data(user)
#                 return HttpResponse(json.dumps({"status": "Success", "user_details": user_details}),
#                                     content_type="application/json")
#             else:
#                 return HttpResponse(str(form.errors))
#         except Exception as e:
#             return HttpResponse(str(e))
#     return HttpResponse("Registration Failed")




# # Database connection
# @csrf_exempt
# def connection(request):
#     global connection_obj
#     if request.method == 'POST':
#         username = request.POST['username']
#         password = request.POST['password']
#         database = request.POST['database']
#         host = request.POST['host']
#         port = request.POST['port']
#         connection_obj = db.create_connection(username, password, database, host, port)
#         print(connection_obj)
#         return HttpResponse(json.dumps({"tables": connection_obj}), content_type="application/json")




# Upload data to the database
# Upload data to the database (CSV and Excel)
@csrf_exempt
def upload_data(request):
    if request.method == 'POST':
        files = request.FILES.get('file')
        if not files:
            return HttpResponse('No files uploaded', status=400)

        file_name = files.name
        file_extension = os.path.splitext(file_name)[1].lower()

        try:
            if file_extension == '.csv':
                content = files.read().decode('utf-8')
                csv_data = io.StringIO(content)
                df = pd.read_csv(csv_data)
            elif file_extension in ['.xls', '.xlsx']:
                df = pd.read_excel(files)
            else:
                raise SuspiciousOperation("Unsupported file format")

            result = db.insert(df, file_name)
            return HttpResponse(result)
        except Exception as e:
            print(e)
            return HttpResponse(f"Failed to upload file: {str(e)}", status=500)


# Showing the number of tables in the database
@csrf_exempt
def get_tableinfo(request):
    if request.method == 'POST':
        table_info = db.get_tables_info()
        return HttpResponse(table_info, content_type="application/json")

# Showing the data to the user based on the table name
@csrf_exempt
def read_data(request):
    if request.method == 'POST':
        tablename = request.POST['tablename']
        df = db.get_table_data(tablename)
        return HttpResponse(df.to_json(), content_type="application/json")

# Data analysis and description of the dataset
@csrf_exempt
def genAIPrompt3(request):
    if request.method == "POST":
        tablename = request.POST.get('tablename', 'data')  # Default to 'data' if not provided
        df = db.get_table_data(tablename)

        # Generate descriptions and questions for the data
        prompt_eng = f"You are analytics_bot. Analyse the data: {df} and give description of the columns. Give the data in the structured format"
        code = generate_code(prompt_eng)

        prompt_eng1 = f"Generate 10 questions for the data: {df}. Give the questions in the linewise format"
        code1 = generate_code(prompt_eng1)

        prompt_eng_2 = f"Generate 10 simple possible plotting questions for the data: {df}. Give the plotting Questions in the linewise format"
        code2 = generate_code(prompt_eng_2)

        # Create a JSON response with titles corresponding to each prompt
        response_data = {
            "column_description": code,
            "text_questions": code1,
            "plotting_questions": code2
        }

        # Print the responses with corresponding titles for debugging purposes
        print("Column Description:")
        print(code)
        print("\nText Questions:")
        print(code1)
        print("\nPlotting Questions:")
        print(code2)

        # Return the response in JSON format
        return JsonResponse(response_data)

    return HttpResponse("Invalid Request Method", status=405)

# Data Visualization and getting the graphs


@csrf_exempt
def genAIPrompt(request):
    if request.method == "POST":
        tablename = request.POST.get('tablename', 'data')  # Default to 'data' if not provided
        df = db.get_table_data(tablename)

        csv_metadata = {"columns": df.columns.tolist()}
        metadata_str = ", ".join(csv_metadata["columns"])
        query = request.POST["query"]

        prompt_eng = (
            f"You are an AI specialized in data analysis and visualization. "
            f"The data is in the table '{tablename}' and its attributes are: {metadata_str}. "
            f"If the user asks for a graph, generate only the Python code using Matplotlib to plot the graph, "
            f"including any necessary calculations like mean, median, etc., based on the data. "
            f"Save the graph as 'graph.png'. "
            f"If the user does not ask for a graph, simply answer the query with the computed result. "
            f"The user asks: {query}."
        )

        code = generate_code(prompt_eng)

        if 'import matplotlib' in code:
            try:
                exec(code, {'df': df, 'plt': plt})  # Execute the code with 'df' and 'plt' in the scope
                with open("graph.png", 'rb') as image_file:
                    return HttpResponse(json.dumps({"graph": base64.b64encode(image_file.read()).decode('utf-8')}),
                                        content_type="application/json")
            except Exception as e:
                prompt_eng = (
                    f"There was an error in the previous code: {str(e)}. "
                    f"Please provide the correct full Python code, including error handling."
                )
                code = generate_code(prompt_eng)
                try:
                    exec(code, {'df': df, 'plt': plt})  # Execute the corrected code with 'df' and 'plt' in the scope
                    with open("graph.png", 'rb') as image_file:
                        return HttpResponse(json.dumps({"graph": base64.b64encode(image_file.read()).decode('utf-8')}),
                                            content_type="application/json")
                except Exception as e:
                    return HttpResponse(json.dumps({"error": f"Failed to generate the chart: {str(e)}"}), content_type="application/json")
        else:
            return HttpResponse(json.dumps({"response": code}), content_type="application/json")



# Function to generate code from OpenAI API
def generate_code(prompt_eng):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
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

    code_start = all_text.find("```python") + 9
    code_end = all_text.find("```", code_start)
    code = all_text[code_start:code_end]
    return code


#Predict and forecast purpose
#Getting the prediction result
@csrf_exempt
def get_prediction_info(request, data, field):
    with open(f"data/{data.lower()}/{field}/results.json", 'r') as fp:
        res = json.load(fp)
    return HttpResponse(json.dumps({"data": res}), content_type="application/json")


#This will return the columns for the table
@csrf_exempt
def get_columns(request, data):
    df = pd.read_csv(f"data/{data.lower()}/processed_data.csv")
    cols = set(df.columns) - {"Store ID", "Employee Number", "Area"}
    return HttpResponse(json.dumps({"columns": list(cols)}), content_type="application/json")


#Generating the URL for the prediction
@csrf_exempt
def generate_deployment(request, data, field):
    hash_object = hashlib.sha256(f'{data}__{field}'.encode('ascii'))
    hex_dig = hash_object.hexdigest()
    save_deployments(hex_dig, data, field)
    return HttpResponse(json.dumps({"deployment_url": hex_dig}), content_type="application/json")


#THis will be helpful for the prediction on which columns.
@csrf_exempt
def deployment(request, data):
    model = get_deployment_txt(data)
    data, field = model.split('___')
    with open(f"data/{data.lower()}/{field.replace(' ', '_')}/deployment.json", 'r') as fp:
        data = json.load(fp)
    cols = set(data["columns"]) - {"Date", field, "Area"}
    return HttpResponse(json.dumps({"columns": list(cols)}), content_type="application/json")


#For prediction based on the deployment url
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
        print(result)
        if isinstance(result, np.ndarray):
            result = str(result[0])
        return HttpResponse(json.dumps({"result": result}), content_type="application/json")


#Forecast of the data with the help of the deployment url
@csrf_exempt
def deployment_forecast(request, col):
    if request.method == 'POST':
        col=col.lower().replace(" ","_")
        with open(os.path.join('data', col+'_results.json'),'r') as fp:
            data= json.load(fp)
        return HttpResponse(json.dumps({"result": data}), content_type="application/json")


def load_models(path, prediction_col, df):
    try:
        if 'retail_sales_data' in path:
            print(df)
            model = joblib.load(os.path.join('data', path.lower(), prediction_col.replace(" ", "_"), "model.joblib"))
            res = model.predict(df.iloc[0, :].to_numpy().reshape(1, -1))
            return res
        else:
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




#
# # Create your views here.
#
#
#
#
# # Establish database connection
# @csrf_exempt
# def connection(request):
#     global connection_obj
#     if request.method == 'POST':
#         username = request.POST['username']
#         password = request.POST['password']
#         database = request.POST['database']
#         host = request.POST['host']
#         port = request.POST['port']
#         connection_obj = db.create_connection(username, password, database, host, port)
#     return HttpResponse(json.dumps({"tables": connection_obj}), content_type="application/json")
#
#
# # Upload and store data in the database
# @csrf_exempt
# def upload_data(request):
#     if request.method == "POST":
#         files = request.FILES.getlist('file')
#         if len(files) < 1:
#             return HttpResponse('No files uploaded')
#         else:
#             for file in files:
#                 content = file.read().decode('utf-8')
#                 csv_data = io.StringIO(content)
#                 df = pd.read_csv(csv_data)
#                 # Save data to the database (adjust the table name as needed)
#                 table_name = os.path.splitext(file.name)[0]  # Use filename without extension as table name
#                 db.insert(df, table_name)
#             return HttpResponse("Successlly Inserted into the database.")
#     else:
#         return HttpResponse("Failure")
#
#
# # Read data from the database
# @csrf_exempt
# def read_data(request):
#     if request.method == 'POST':
#         tablename = request.POST['tablename']
#         df = db.get_table_data(tablename)
#         return HttpResponse(df.to_json(orient="records"), content_type="application/json")
#
#
# # Get information about all tables
# @csrf_exempt
# def get_tableinfo(request):
#     if request.method == 'POST':
#         df = db.get_tables_info()
#         return HttpResponse(df, content_type="application/json")
#
# # Get prediction information based on the stored results
# @csrf_exempt
# def get_prediction_info(request, data, field):
#     with open(f"data/{data.lower()}/{field}/results.json", 'r') as fp:
#         res = json.load(fp)
#     return HttpResponse(json.dumps({"data": res}), content_type="application/json")
#
#
# # Get column names of a specific dataset
# @csrf_exempt
# def get_columns(request, data):
#     df = db.get_table_data(data)
#     cols = set(df.columns)
#     return HttpResponse(json.dumps({"columns": list(cols)}), content_type="application/json")
#
#
# # Generate deployment information for a specific dataset and field
# @csrf_exempt
# def generate_deployment(request, data, field):
#     hash_object = hashlib.sha256(f'{data}__{field}'.encode('ascii'))
#     hex_dig = hash_object.hexdigest()
#     save_deployments(hex_dig, data, field)
#     return HttpResponse(json.dumps({"deployment_url": hex_dig}), content_type="application/json")
#
#
# # Retrieve deployment information and available columns
# @csrf_exempt
# def deployment(request, data):
#     model = get_deployment_txt(data)
#     data, field = model.split('___')
#     with open(f"data/{data.lower()}/{field.replace(' ', '_')}/deployment.json", 'r') as fp:
#         data = json.load(fp)
#     cols = set(data["columns"]) - {"Date", field, "Area"}
#     return HttpResponse(json.dumps({"columns": list(cols)}), content_type="application/json")
#
#
# # Predict outcomes based on user input and the deployed model
# @csrf_exempt
# def deployment_predict(request, data):
#     if request.method == 'POST':
#         model = get_deployment_txt(data)
#         data, field = model.split('___')
#         res = {}
#         for col in request.POST:
#             res.update({col: request.POST[col]})
#         df = pd.DataFrame([res])
#         result = load_models(data, field, df)
#         if isinstance(result, np.ndarray):
#             result = str(result[0])
#         return HttpResponse(json.dumps({"result": result}), content_type="application/json")
#
#
# # Forecast outcomes based on stored prediction results
# @csrf_exempt
# def deployment_forecast(request, col):
#     if request.method == 'POST':
#         col = col.lower().replace(" ", "_")
#         with open(os.path.join('data', col + '_results.json'), 'r') as fp:
#             data = json.load(fp)
#         return HttpResponse(json.dumps({"result": data}), content_type="application/json")
#
#
# # Load models dynamically based on the data provided by the user
# def load_models(path, prediction_col, df):
#     try:
#         model_dir = os.path.join('data', path.lower(), prediction_col.replace(" ", "_"))
#
#         with open(os.path.join(model_dir, "deployment.json"), 'r') as fp:
#             deployment_data = json.load(fp)
#
#         model_type = deployment_data["model_type"]
#
#         # Handle preprocessing based on deployment configuration
#         for column, config in deployment_data["columns"].items():
#             if isinstance(config, list):  # Encoding case
#                 encoder_path = os.path.join(model_dir, f'{column.replace(" ", "_")}_encoder.pkl')
#                 encoder = joblib.load(encoder_path)
#                 df[column] = encoder.transform(df[column])
#             else:  # Scaling case or other transformation
#                 df[column] = df[column].astype(float)
#
#         # Load the model
#         model_file = "model.joblib" if model_type in ["classification", "regression"] else "model.h5"
#         model_path = os.path.join(model_dir, model_file)
#
#         if model_file.endswith('.joblib'):
#             model = joblib.load(model_path)
#         else:
#             model = load_model(model_path)
#
#         # Make predictions
#         input_data = df.iloc[0, :].to_numpy().reshape(1, -1)
#         res = model.predict(input_data)
#
#         # Post-process predictions based on the model type
#         if model_type == 'classification':
#             result = np.argmax(res, axis=-1)
#             encoder = joblib.load(os.path.join(model_dir, f'{prediction_col.replace(" ", "_")}_encoder.pkl'))
#             res = encoder.inverse_transform(result)
#             return res[0]
#         else:
#             # For regression or other types, directly return the result
#             return res[0] if isinstance(res, np.ndarray) else res
#
#     except Exception as e:
#         print(f"An error occurred while loading the model: {e}")
#         return None
#
#
# # Save deployment information for later retrieval
# def save_deployments(hex_data, data, field):
#     if not os.path.exists("deployments.json"):
#         deployment_data = {}
#     else:
#         with open("deployments.json", 'r') as fp:
#             deployment_data = json.load(fp)
#     deployment_data.update({hex_data: f'{data}___{field.replace(" ", "_")}'})
#     with open("deployments.json", 'w') as fp:
#         json.dump(deployment_data, fp)
#
#
# # Retrieve deployment information based on the unique hash
# def get_deployment_txt(hex_data):
#     with open("deployments.json", 'r') as fp:
#         deployment_data = json.load(fp)
#     return deployment_data[hex_data]
