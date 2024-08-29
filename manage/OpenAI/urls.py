from django.urls import path
from .views import *

urlpatterns = [
    #path("api/", testing, name="testing"),
   # path(r'api/register', register, name='register'),
   # path(r'api/login', loginpage, name='login'),
   # path('api/connect', connection, name='connection'),
    path("api/upload", upload_data, name='upload_data'),
    path('api/tableinfo', get_tableinfo, name='get_table_info'),
    path("api/tabledata", read_data, name='read_table_data'),
    path("api/getAnalytics", genAIPrompt3, name="GenAIPrompt3"),
    path("api/getResult", genAIPrompt, name="GenAIPrompt"),
    # forecast/predict apis
    path('api/predict/<str:data>/<str:field>', get_prediction_info, name='get_prediction_info'),
    path('api/predict/<str:data>', get_columns, name='get_columns'),
    path('api/deployments/<str:data>', deployment, name='deployment'),
    path('api/generatedeployment/<str:data>/<str:field>', generate_deployment, name='generate_deployment'),
    path('api/deployments/<str:data>/predict', deployment_predict, name='deployment_predict'),
    path('api/forecast/<str:col>', deployment_forecast, name='deployment_forecast')
]
