import base64
import io
import json
import os
import sys
from io import StringIO
import pandas as pd
import razorpay
from django.conf import settings
from django.contrib.auth import authenticate, login, logout
from django.http import HttpResponse
from django.http import HttpResponseBadRequest
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from dotenv import load_dotenv
from openai import OpenAI
from django.http import JsonResponse
from digiotai.digiotai_jazz import Agent, Task, OpenAIModel, SequentialFlow, InputType, OutputType
from .database import SQLiteDB
from .form import CreateUserForm
import base64
import jwt


razorpay_client = razorpay.Client(
    auth=(settings.RAZOR_KEY_ID, settings.RAZOR_KEY_SECRET))

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configure OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)
expertise = "Interior Desinger"
task = Task("Image Generation")
input_type = InputType("Text")
output_type = OutputType("Image")
agent = Agent(expertise, task, input_type, output_type)
api_key = OPENAI_API_KEY
jwt_secret = "my_sampe_token"
db = SQLiteDB()
# db.table_creation()
user_name = None


@csrf_exempt
def testing(request):
    return HttpResponse("Application is up")


def get_csv_metadata(df):
    metadata = {
        "columns": df.columns.tolist(),
        "data_types": df.dtypes.to_dict(),
        "null_values": df.isnull().sum().to_dict(),
        "example_data": df.head().to_dict()
    }
    return metadata

# This function will be used for the login purpose
@csrf_exempt
def loginpage(request):
    """ if user submits the credentials  then it check if they are valid or not
                    if it is valid then it redirects to user home page """
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            token = jwt.encode(payload={
                "sub":'log',
                "usenamr": username,
                "nickname":'test'
            },
            key=jwt_secret)
            return HttpResponse(token)
        else:
            print('User Name or Password is incorrect')
            return HttpResponse('User Name or Password is incorrect')
    context = {}
    return HttpResponse("Login failed")

# This function will be used for the logout from the application
@csrf_exempt
# logout method
def logoutpage(request):
    try:
        logout(request)
        request.session.clear()   # deleting the session of user
        return redirect('demo:login')  # redirecting to login page
    except Exception as e:
        return e  # redirect('demo:')  # redirecting to login page


# This function will be used for the user registration
@csrf_exempt
def register(request):
    if request.method == 'POST':
        form = CreateUserForm(request.POST)
        try:
            if form.is_valid():
                form.save()
                user = form.cleaned_data.get('username')
                email = form.cleaned_data.get('email')
                db.add_user(user, email)
                token = jwt.encode(payload={
                    "sub": 'log',
                    "usenamr": user,
                    "nickname": 'test'
                },
                    key=jwt_secret)
                return HttpResponse(token)

            else:
                print(form.errors)
                return HttpResponse(str(form.errors))
        except Exception as e:
            print(e)

    return HttpResponse("Registration Failed1")


# This will be used for the login through google
@csrf_exempt
def googlelogin(request):
    username = request.POST.get("username")
    password = username + "@" + request.POST.get("id")
    email = request.POST.get("email")
    users = db.get_users()
    token = jwt.encode(payload={
        "sub": 'log',
        "usename": username,
        "nickname": 'test'
    },
        key=jwt_secret)
    if username in users:
        return HttpResponse(token)
    else:
        form = CreateUserForm({'username': username,'email': email, 'password1': password, 'password2': password})
        if form.is_valid():
            form.save()
            user = form.cleaned_data.get('username')
            email = form.cleaned_data.get('email')
            db.add_user(user, email)
            return HttpResponse(token)
        else:
            print(form.errors)
            return HttpResponse(str(form.errors))


# @csrf_exempt
# def upload_data(request):
#     if request.method == "POST":
#         files = request.FILES['file']
#         if len(files) < 1:
#             return HttpResponse('No files uploaded')
#         else:
#             content = files.read().decode('utf-8')
#             csv_data = io.StringIO(content)
#             df = pd.read_csv(csv_data)
#             df.to_csv('data.csv', index=False)
#             result = genAIPrompt3(user_name = jwt.decode(request.headers['token'],key=jwt_secret,algorithms=["HS256",])["usename"])
#             return HttpResponse(json.dumps(result),
#                                 content_type="application/json")
#     else:
#         return HttpResponse("Failure")

# # This is for generating the chart
# @csrf_exempt
# def genAIPrompt(request):
#     if request.method == "POST":
#         df = pd.read_csv("data.csv")
#         csv_metadata = get_csv_metadata(df)
#         metadata_str = ", ".join(csv_metadata["columns"])
#         query = request.POST["query"]
#         prompt_eng = (
#             f"You are graphbot. If the user asks to plot a graph, you only reply with the Python code of Matplotlib to plot the graph and save it as graph.png. "
#             f"The data is in data.csv and its attributes are: {metadata_str}. If the user does not ask for a graph, you only reply with the answer to the query. "
#             f"The user asks: {query}"
#         )

#         code = generate_code(prompt_eng)
#         if 'import matplotlib' in code:
#             try:
#                 exec(code)
#                 with open("graph.png", 'rb') as image_file:
#                     return HttpResponse(json.dumps({"graph": base64.b64encode(image_file.read()).decode('utf-8')}),
#                                         content_type="application/json")
#             except Exception as e:
#                 prompt_eng = f"There has occurred an error while executing the code, please take a look at the error and strictly only reply with the full python code do not apologize or anything just give the code {str(e)}"
#                 code = generate_code(prompt_eng)
#                 try:
#                     exec(code)
#                     with open("graph.png", 'rb') as image_file:
#                         return HttpResponse(json.dumps({"graph": base64.b64encode(image_file.read()).decode('utf-8')}),
#                                             content_type="application/json")
#                 except Exception as e:
#                     return HttpResponse("Failed to generate the chart. Please try again")
#         else:
#             return HttpResponse(code)



# This is for the generation of the image from the text
@csrf_exempt
def genAIPrompt2(request):
    if request.method == "POST":
        model = OpenAIModel(api_key=api_key, model="dall-e-2")
        sequential_flow = SequentialFlow(agent, model)
        selected_style = request.POST["selected_style"]
        selected_room_color = request.POST["selected_room_color"]
        selected_room_type = request.POST["selected_room_type"]
        number_of_room_designs = request.POST["number_of_room_designs"]
        additional_instructions = request.POST["additional_instructions"]
        username = request.POST["username"]
        stat, count, quota = checkQuota(username)
        if stat:
            prompt = f"Generate a Realistic looking Interior design with the following instructions: style: {selected_style}, Room Color: {selected_room_color}, Room type: {selected_room_type}, Number of designs: {number_of_room_designs}, Instructions: {additional_instructions}"
            image_url = sequential_flow.execute(prompt)
            print(image_url)
            if quota == "FREE":
                db.update_count(username)
                count -= 1
            return HttpResponse(json.dumps({"image": image_url, "status": "Success", "count": count}),
                                content_type="application/json")
        else:
            return HttpResponse(json.dumps({"image": "NA", "status": "Quota limit exceeded", "count": count}),
                                content_type="application/json")





# This will be used for generationg the code from the prompt
# @csrf_exempt
# def generate_code(prompt_eng):
#     response = client.chat.completions.create(
#         model="gpt-4-turbo",
#         messages=[
#             {"role": "system", "content": "You are a helpful assistant."},
#             {"role": "user", "content": prompt_eng}
#         ]
#     )
#     all_text = ""
#     # Display generated content dynamically
#     for choice in response.choices:
#         print(f"Debug - choice structure: {choice}")  # Debugging line
#         message = choice.message
#         print(f"Debug - message structure: {message}")  # Debugging line
#         chunk_message = message.content if message else ''
#         all_text += chunk_message

#     code_start = all_text.find("```python") + 9
#     code_end = all_text.find("```", code_start)
#     code = all_text[code_start:code_end]
#     return code


# This will be used for the analysing data.
# @csrf_exempt
# def genAIPrompt3(user):
#     df = pd.read_csv("data.csv")
#     prompt_eng = (
#         f"You are analytics_bot. Analyse the data: {df} and give description of the columns"
#     )
#     if checkQuota(user):
#         code = generate_code(prompt_eng)
#         prompt_eng1 = (
#             f"Generate 10 questions for the data: {df}"
#         )

#         prompt_eng_2 = f"Generate 10 simple possible plotting questions for the data: {df}"

#         code1 = generate_code(prompt_eng1)

#         code2 = generate_code(prompt_eng_2)
#         db.update_count(user)
#         return {"status": "Success", "col_desc": code, "sample data": df.head(10).to_json(), "text_questions": code1,
#                 "chart_questions": code2}
#     else:
#         return {"status": 'Quota limit exceeded'}


def checkQuota(user):
    user_details = db.get_user_data(user)
    quota = user_details[1]
    count = user_details[2]
    if quota != 'FREE':
        return True, count, quota
    else:
        if 0 < count <= 10:
            return True, count, quota
        else:
            return False, count, quota

# # Helper function for analysinig data
# @csrf_exempt
# def regenerate_txt(request):
#     df = pd.read_csv("data.csv")
#     prompt_eng = (
#         f"Regenerate 10 questions for the data: {df}"
#     )
#     code = generate_code(prompt_eng)
#     return HttpResponse(json.dumps({"questions": code}),
#                         content_type="application/json")

# # For the regenration of the chart
# @csrf_exempt
# def regenerate_chart(request):
#     df = pd.read_csv("data.csv")
#     prompt_eng = (
#         f"Regenerate 10 simple possible plotting questions for the data: {df}. start the question using plot keyword"
#     )
#     code = generate_code(prompt_eng)
#     return HttpResponse(json.dumps({"questions": code}),
#                         content_type="application/json")


# ## Helper Function for the generating code
# @csrf_exempt
# def genresponse(request):
#     df = pd.read_csv("data.csv")
#     if request.method == "POST":
#         question = request.POST["query"]
#         graph = ''
#         if os.path.exists("graph.png"):
#             os.remove("graph.png")
#         prompt_eng = (
#             f"generate python code for the question {question} based on the data: {df} from data.csv file. "
#             f"If the question is related to plotting then save the plot as graph.csv"
#         )
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

#             return HttpResponse(json.dumps({"answer": redirected_output.getvalue(), "graph": graph}),
#                                 content_type="application/json")
#         return HttpResponse(json.dumps({"answer": code}),
#                             content_type="application/json")



# This function will be used for the payment purpose
@csrf_exempt
def paymentinfo(request, ):
    if request.method == "POST":
        currency = 'INR'
        amount = request.POST['amount']  # Rs. 200
        request.session['amount'] = amount
        # Create a Razorpay Order
        razorpay_order = razorpay_client.order.create(dict(amount=amount,
                                                           currency=currency,
                                                           payment_capture='0'))
        # order id of newly created order.
        razorpay_order_id = razorpay_order['id']
        callback_url = 'http://3.132.248.171:4500/paymenthandler/'

        # we need to pass these details to frontend.
        context = {}
        context['razorpay_order_id'] = razorpay_order_id
        context['razorpay_merchant_key'] = settings.RAZOR_KEY_ID
        context['razorpay_amount'] = amount
        context['currency'] = currency
        context['callback_url'] = callback_url

        return HttpResponse(json.dumps({"paymentinfo": context}),
                            content_type="application/json")



# Payment handling function
# we need to csrf_exempt this url as
# POST request will be made by Razorpay
# and it won't have the csrf token.
@csrf_exempt
def paymenthandler(request):
    # only accept POST request.
    if request.method == "POST":
        if "razorpay_signature" in request.POST:
            payment_verification = razorpay_client.utility.verify_payment_signature(request.POST)
            if payment_verification:
                return JsonResponse({"res": "success"})
                # Logic to perform is payment is successful
            else:
                return JsonResponse({"res": "failed"})
        else:
            return HttpResponse("Signature not available, payment failed")
    else:
        return HttpResponse("Get not valid")


@csrf_exempt
def get_user_details(request):
    if request.method == "POST":
        return HttpResponse(json.dumps({"paymentinfo": list(db.get_user_data(request.POST.get("user")))}),
                            content_type="application/json")


@csrf_exempt
def updateuserplan(request):
    if request.method == "POST":
        user = request.POST.get("user")
        plan = request.POST.get("plan")
        data = db.get_user_data(user)
        if plan.upper() == "BASIC":
            c = data[2] + 50
        elif plan.upper() == "PRO":
            c = data[2] + 200
        else:
            c = data[0]
        db.update_user(user, plan, c)
        return get_user_details(request)


# This is for the conversion of the three images to the single image.
@csrf_exempt
def generateImage(request):
    if request.method == "POST":
        model = OpenAIModel(api_key=api_key, model="dall-e-2")
        sequential_flow = SequentialFlow(agent, model)
        selected_style = request.FILES["selected_style"]
        selected_room_color = request.FILES["selected_room_color"]
        selected_room_type = request.FILES["selected_room_type"]
        s_style = base64.b64encode(selected_style.read()).decode('utf-8')
        s_room_c = base64.b64encode(selected_room_color.read()).decode('utf-8')
        s_room_t = base64.b64encode(selected_room_type.read()).decode('utf-8')
        user_name = jwt.decode(request.headers['token'],key=jwt_secret,algorithms=["HS256",])["usename"]
        stat, count, quota = checkQuota(user_name)
        if stat:
            prompt = f"Generate a final Image based on the 3 input images provided: Image_1={s_style}, Image_2={s_room_c}, " \
                     f"Image_3={s_room_t}"
            image_url = sequential_flow.execute(prompt)
            if quota == "FREE":
                db.update_count(user_name)
                count -= 1
            return HttpResponse(json.dumps({"image": image_url, "status": "Success", "count": count}),
                                content_type="application/json")
        else:
            return HttpResponse(json.dumps({"image": "NA", "status": "Quota limit exceeded", "count": count}),
                                content_type="application/json")


# @csrf_exempt
# def generatetestImage(request):
#     if request.method == "POST":
#         model = OpenAIModel(api_key=api_key, model="dall-e-2")
#         sequential_flow = SequentialFlow(agent, model)
#         prompt = request.POST.get("prompt")
#         prompt = f"Generate an image based on user prompt and save it as graph1.png. User prompt is {prompt}"
#         image_url = sequential_flow.execute(prompt)

#         return HttpResponse(json.dumps({"image": image_url, "status": "Success"}),
#                             content_type="application/json")
