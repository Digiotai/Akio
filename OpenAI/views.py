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




# Database connection
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
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
import os
import io
import pandas as pd
import json
from django.core.exceptions import SuspiciousOperation

@csrf_exempt
def upload_and_analyze_data(request):
    if request.method == 'POST':
        # Handle file upload
        files = request.FILES.get('file')
        if not files:
            return HttpResponse('No files uploaded', status=400)

        file_name = files.name
        file_extension = os.path.splitext(file_name)[1].lower()

        try:
            # Process the uploaded file
            if file_extension == '.csv':
                content = files.read().decode('utf-8')
                csv_data = io.StringIO(content)
                df = pd.read_csv(csv_data)
            elif file_extension in ['.xls', '.xlsx']:
                df = pd.read_excel(files)
            else:
                raise SuspiciousOperation("Unsupported file format")

            # Insert the data into the database
            result = db.insert(df, file_name)  # Assuming db.insert returns the DataFrame

            # Extract the first 10 rows of the data
            first_10_rows = df.head(10).to_dict(orient='records')

            # Generate descriptions and questions for the data based on table name and columns
            columns_list = ", ".join(df.columns)

            prompt_eng = (
                f"You are an AI specializing in data analysis. The table '{file_name}' has the following columns: {columns_list}. "
                f"Provide a detailed description of each column in the table."
            )
            column_description = generate_code(prompt_eng)

            prompt_eng1 = (
                f"Based on the table '{file_name}' and its columns: {columns_list}, "
                f"generate 10 simple  and very basic questions that are related to data preprocessing steps and  calculating basic statistics such as averages, sums, or counts based on the given tablename only. "
                f"Do not include any questions related to graph plotting or visualization. "
                f"Provide these questions in a linewise format."
            )
            text_questions = generate_code(prompt_eng1)

            prompt_eng2 = (
                f"Based on the table '{file_name}' and its columns: {columns_list}, "
                f"generate 10 simple questions that focus on graph plotting and visualizations, such as creating histograms, scatter plots, bar charts, or line graphs. "
                f"These questions should be related to visualizing patterns, trends, distributions, or comparisons in the data. "
                f"Do not include any questions about data preprocessing or analysis. "
                f"Provide these questions in a linewise format."
            )
            plotting_questions = generate_code(prompt_eng2)

            # Create a JSON response with titles corresponding to each prompt
            response_data = {
                "first_10_rows": first_10_rows,  # Include first 10 rows
                "column_description": column_description,
                "text_questions": text_questions,
                "plotting_questions": plotting_questions
            }

            # Return the analytics response along with the first 10 rows
            return JsonResponse(response_data, safe=False)

        except Exception as e:
            print(e)
            return HttpResponse(f"Failed to upload and analyze file: {str(e)}", status=500)

    return HttpResponse("Invalid Request Method", status=405)


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


# Data Visualization and getting the graphs


# @csrf_exempt
# def genAIPrompt(request):
#     if request.method == "POST":
#         tablename = request.POST.get('tablename', 'data')  # Default to 'data' if not provided
#         df = db.get_table_data(tablename)
#
#         csv_metadata = {"columns": df.columns.tolist()}
#         metadata_str = ", ".join(csv_metadata["columns"])
#         query = request.POST["query"]
#
#         prompt_eng = (
#             f"You are an AI specialized in data analysis and visualization. "
#             f"The data is in the table '{tablename}' and its attributes are: {metadata_str}. "
#             f"If the user asks for a graph, generate only the Python code using Matplotlib to plot the graph, "
#             f"including any necessary calculations like mean, median, etc., based on the data. "
#             f"Save the graph as 'graph.png'. "
#             f"If the user does not ask for a graph, simply answer the query with the computed result. "
#             f"The user asks: {query}."
#         )
#
#         code = generate_code(prompt_eng)
#
#         if 'import matplotlib' in code:
#             try:
#                 exec(code, {'df': df, 'plt': plt})  # Execute the code with 'df' and 'plt' in the scope
#                 with open("graph.png", 'rb') as image_file:
#                     return HttpResponse(json.dumps({"graph": base64.b64encode(image_file.read()).decode('utf-8')}),
#                                         content_type="application/json")
#             except Exception as e:
#                 prompt_eng = (
#                     f"There was an error in the previous code: {str(e)}. "
#                     f"Please provide the correct full Python code, including error handling."
#                 )
#                 code = generate_code(prompt_eng)
#                 try:
#                     exec(code, {'df': df, 'plt': plt})  # Execute the corrected code with 'df' and 'plt' in the scope
#                     with open("graph.png", 'rb') as image_file:
#                         return HttpResponse(json.dumps({"graph": base64.b64encode(image_file.read()).decode('utf-8')}),
#                                             content_type="application/json")
#                 except Exception as e:
#                     return HttpResponse(json.dumps({"error": f"Failed to generate the chart: {str(e)}"}), content_type="application/json")
#         else:
#             return HttpResponse(json.dumps({"response": code}), content_type="application/json")



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

    code_start = all_text.find("```python") + 9
    code_end = all_text.find("```", code_start)
    code = all_text[code_start:code_end]
    return code



@csrf_exempt
def regenerate_txt(request):
    if request.method == "POST":
        tablename = request.POST.get('tablename', 'data')  # Default to 'data' if not provided
        df = db.get_table_data(tablename)
        prompt_eng = (
             f"Regenerate 10 questions for the data: {df}"
    )
        code = generate_code(prompt_eng)
        return HttpResponse(json.dumps({"questions": code}),
                        content_type="application/json")

@csrf_exempt
def regenerate_chart(request):
    if request.method == "POST":
        tablename = request.POST.get('tablename', 'data')  # Default to 'data' if not provided
        df = db.get_table_data(tablename)
        prompt_eng = (
            f"Regenerate 10 simple possible plotting questions for the data: {df}. start the question using plot keyword"
    )
        code = generate_code(prompt_eng)
        return HttpResponse(json.dumps({"questions": code}),
                        content_type="application/json")


#
# @csrf_exempt
# def genresponse(request):
#     if request.method == "POST":
#         tablename = request.POST.get('tablename', 'data')  # Default to 'data' if not provided
#         df = db.get_table_data(tablename)
#
#         csv_metadata = {"columns": df.columns.tolist()}
#         metadata_str = ", ".join(csv_metadata["columns"])
#         query = request.POST["query"]
#
#         graph = ''
#         if os.path.exists("graph.png"):
#             os.remove("graph.png")
#
#         prompt_eng = (
#             f"You are an AI specialized in data analysis and visualization. "
#             f"The data is in the table '{tablename}' and its attributes are: {metadata_str}. "
#             f"If the user asks for a graph, generate only the Python code using Matplotlib to plot the graph, "
#             f"including any necessary calculations like mean, median, etc., based on the data. "
#             f"Save the graph as 'graph.png'. "
#             f"If the user does not ask for a graph, simply answer the query with the computed result. "
#             f"The user asks: {query}."
#
#         )
#
#         code = generate_code(prompt_eng)
#         if "import" in code:
#             old_stdout = sys.stdout
#             redirected_output = sys.stdout = StringIO()
#             exec(code)
#             sys.stdout = old_stdout
#             print(redirected_output.getvalue())
#             if os.path.exists("graph.png"):
#                 with open("graph.png", 'rb') as image_file:
#                     graph = base64.b64encode(image_file.read()).decode('utf-8')
#
#             return HttpResponse(json.dumps({"answer": redirected_output.getvalue(), "graph": graph}),
#                                 content_type="application/json")
#         return HttpResponse(json.dumps({"answer": code}),
#                             content_type="application/json")



# from django.http import JsonResponse, HttpResponse
# from django.views.decorators.csrf import csrf_exempt
# import sys
# from io import StringIO

# @csrf_exempt
# def genresponse(request):
#     if request.method == "POST":
#         tablename = request.POST.get('tablename', 'data')  # Default to 'data' if not provided
#         df = db.get_table_data(tablename)
#
#         csv_metadata = {"columns": df.columns.tolist()}
#         metadata_str = ", ".join(csv_metadata["columns"])
#         query = request.POST["query"]
#
#         prompt_eng = (
#             f"generate python code for the question {query} based on the data: {df} from {tablename} and columns are{metadata_str}. "
#             f"Do not generate any graph or Python code for plotting. Just provide the computed result."
#         )
#
#         code = generate_code(prompt_eng)
#         return HttpResponse(json.dumps({"answer": code}), content_type="application/json")
#
#     return HttpResponse("Invalid Request Method", status=405)



#For genai

import json
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
import io
import sys

@csrf_exempt
def genresponse(request):
    if request.method == "POST":
        tablename = request.POST.get('tablename', "data")  # Default to 'data' if not provided
        df = db.get_table_data(tablename)
        print(df)

        # Save the data to a CSV file
        csv_file_path = 'data1.csv'
        df.to_csv(csv_file_path, index=False)

        # Generate CSV metadata
        csv_metadata = {"columns": df.columns.tolist()}
        metadata_str = ", ".join(csv_metadata["columns"])

        query = request.POST["query"]

        print("execution started")

        prompt_eng = (
            f"You are an AI specialized in data preprocessing."
            f"Data related to the {query} is stored in a CSV file data1.csv.Consider the data1.csv as the data source"
            f"Generate Python code to answer the question: '{query}' based on the data from '{tablename}'. "
            f"The DataFrame 'df' contains the following columns: {metadata_str}. "
            f"Return only the Python code that computes the result .Result should describe the parameters in it, without any plotting or visualization."
            f"If the {query} related to the theoretical concept.You will give a small description about the concept also."


        )

        code = generate_code(prompt_eng)

        print(code)

        # Execute the generated code
        result = execute_code(code, df)

        return JsonResponse({"answer": result})

    return HttpResponse("Invalid Request Method", status=405)


def execute_code(code, df):
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
import os
import json
import pandas as pd
from django.http import HttpResponse, FileResponse
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def genAIPrompt(request):
    if request.method == "POST":
        tablename = request.POST.get('tablename', "data")  # Default to 'data' if not provided
        df = db.get_table_data(tablename)
        print(df)

        # Save the data to a CSV file
        csv_file_path = 'data.csv'
        df.to_csv(csv_file_path, index=False)

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
            f"Save the graph as 'graph.png'. "
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


#For genbi
import json
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
import io
import sys

@csrf_exempt
def genresponse2(request):
    if request.method == "POST":
        tablename = request.POST.get('tablename', "data")  # Default to 'data' if not provided
        df = db.get_table_data(tablename)
        print(df)

        # Save the data to a CSV file
        csv_file_path = 'data2.csv'
        df.to_csv(csv_file_path, index=False)

        # Generate CSV metadata
        csv_metadata = {"columns": df.columns.tolist()}
        metadata_str = ", ".join(csv_metadata["columns"])

        query = request.POST["query"]

        print("execution started")

        prompt_eng = (
            f"You are an AI specialized in data preprocessing."
            f"Data related to the {query} is stored in a CSV file data2.csv.Consider the data1.csv as the data source"
            f"Generate Python code to answer the question: '{query}' based on the data from '{tablename}'. "
            f"The DataFrame 'df' contains the following columns: {metadata_str}. "
            f"Return only the Python code that computes the result .Result should describe the parameters in it, without any plotting or visualization."
            f"If the {query} related to the theoretical concept.You will give a small description about the concept also."


        )

        code = generate_code(prompt_eng)

        print(code)

        # Execute the generated code
        result = execute_code(code, df)

        return JsonResponse({"answer": result})

    return HttpResponse("Invalid Request Method", status=405)


def execute_code(code, df):
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
import os
import json
import pandas as pd
from django.http import HttpResponse, FileResponse
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def genAIPrompt2(request):
    if request.method == "POST":
        tablename = request.POST.get('tablename', "data")  # Default to 'data' if not provided
        df = db.get_table_data(tablename)
        print(df)

        # Save the data to a CSV file
        csv_file_path = 'data3.csv'
        df.to_csv(csv_file_path, index=False)

        # Generate CSV metadata
        csv_metadata = {"columns": df.columns.tolist()}
        metadata_str = ", ".join(csv_metadata["columns"])

        query = request.POST["query"]

        prompt_eng = (
            f"You are an AI specialized in data analytics and visualization. "
            f" Data used for analysis is stored in a CSV file data3.csv. "
            f"Attributes of the data are: {metadata_str}. "
            f"Consider 'data3.csv' as the data source for any analysis."
            f"If the user asks for a graph, generate only the Python code using Matplotlib to plot the graph. "
            f"Save the graph as 'graph.png'. "
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
        print(model)
        data, field = model.split('___')
        print(data, field)
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
        # if 'retail_sales_data' in path:
        #     print(df)
        #     model = joblib.load(os.path.join('data', path.lower(), prediction_col.replace(" ", "_"), "model.joblib"))
        #     res = model.predict(df.iloc[0, :].to_numpy().reshape(1, -1))
        #     return res
        # else:
        print("Hai")
        model = load_model(os.path.join('data', path.lower(), prediction_col.replace(" ", "_"), "model.h5"))
        with open(os.path.join('data', path.lower(), prediction_col.replace(" ", "_"), "deployment.json"), 'r') as fp:
            deployment_data = json.load(fp)
        print(df)
        for column in deployment_data["columns"]:
            if isinstance(deployment_data["columns"][column], list):
                encoder_path = os.path.join('data', path.lower(), prediction_col.replace(" ", "_"),
                                            f'{column.replace(" ", "_")}_encoder.pkl')
                df[column.replace("_", " ")] = joblib.load(encoder_path).fit_transform(df[column.replace("_", " ")])
            else:
                df[column] = df[column].astype(float)
        print(df)
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



