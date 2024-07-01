from django.urls import path
from .views import *

urlpatterns = [
    path("", testing, name="testing"),
    path(r'login', loginpage, name='login'),
    path(r'register', register, name='register'),
    #path("upload", upload_data, name="upload_data"),
    #path("getResult", genAIPrompt, name="GenAIPrompt"),
    path("getImage", genAIPrompt2, name="GenAIPrompt2"),
   # path("getAnalytics", genAIPrompt3, name="GenAIPrompt3"),
   # path("regenerate", regenerate_txt, name="regenerate"),
   # path("regenerate_chart", regenerate_chart, name="regenerate_chart"),
    #path("genresponse", genresponse, name="answer"),
    path('paymenthandler/', paymenthandler, name='paymenthandler'),
    path('paymentinfo', paymentinfo, name='paymentinfo'),
    path("get_user_details", get_user_details, name="get_user_details"),
    path("updateuserplan", updateuserplan, name="updateuserplan"),
    path("googlelogin", googlelogin, name="googlelogin"),
    path("generateImage", generateImage, name='generateImage'),
    #path("generatetestImage", generatetestImage, name='generatetestImage'),
]
