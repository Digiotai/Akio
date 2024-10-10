from django.urls import path
from .views import *

urlpatterns = [
    #path("api/", testing, name="testing"),
   # path(r'api/register', register, name='register'),
   # path(r'api/login', loginpage, name='login'),
    path('api/connect', connection, name='connection'),
    path("api/upload", upload_and_analyze_data, name='upload_data'),
    path('api/tableinfo', get_tableinfo, name='get_table_info'),
    path("api/read_db_table_data", read_db_table_data, name='read_table_data'),

    # Analytical
    path("api/gen_txt_response", gen_txt_response, name="regenerate_chart"),
    path("api/gen_graph_response", gen_graph_response, name="GenAIPrompt"),
    path("api/regenerate_txt_questions", regenerate_txt, name="regenerate"),
    path("api/regenerate_graph_questions", regenerate_chart, name="regenerate_chart"),


    # forecast/predict apis
    path('api/predict/<str:data>/<str:field>', get_prediction_info, name='get_prediction_info'),
    path('api/predict/<str:data>', get_columns, name='get_columns'),
    path('api/deployments/<str:data>', deployment, name='deployment'),
    path('api/generatedeployment/<str:data>/<str:field>', generate_deployment, name='generate_deployment'),
    path('api/deployments/<str:data>/predict', deployment_predict, name='deployment_predict'),
    path('api/forecast/<str:data>/<str:col>', deployment_forecast, name='deployment_forecast')
]
