from django.urls import path
from .views import *

urlpatterns = [
    #path("api/", testing, name="testing"),
   # path(r'api/register', register, name='register'),
   # path(r'api/login', loginpage, name='login'),
    path('api/connect', connection, name='connection'),
    path("api/upload", upload_and_analyze_data, name='upload_data'),
    path('api/tableinfo', get_tableinfo, name='get_table_info'),
    path("api/tabledata", read_data, name='read_table_data'),
    path("api/read_db_table_data", read_db_table_data, name='read_table_data'),
    path("api/get_user_data", get_user_data, name='gwt_user_data'),

    # Analytical
    path("api/gen_txt_response", gen_txt_response, name="regenerate_chart"),
    path("api/gen_graph_response", gen_graph_response, name="GenAIPrompt"),
    path("api/regenerate_txt_questions", regenerate_txt, name="regenerate"),
    path("api/regenerate_graph_questions", regenerate_chart, name="regenerate_chart"),
    path("api/regenerate_forecast_questions", regenerate_forecast, name="regenerate_forecast"),
    path('api/get_description', get_description, name='get_description'),

    # train
    path("api/train/<train_type>/<file_name>", train_data, name="train"),

    # forecast/predict apis
    path('api/predict/<str:data>/<str:field>', get_prediction_info, name='get_prediction_info'),
    path('api/<str:train_type>/<str:data>', get_columns, name='get_columns'),
    path('api/deployments/<str:data>', deployment, name='deployment'),
    path('api/generatedeployment/<str:data>/<str:field>', generate_deployment, name='generate_deployment'),
    path('api/deployments/<str:data>/predict', deployment_predict, name='deployment_predict'),
    path('api/forecast/<str:data>/<str:col>', deployment_forecast, name='deployment_forecast'),

    #Forecast with wyge
    path('api/forecasts', forecast_sales, name='forecasting'),
    path('api/synthetic_data', handle_synthetic_data_api, name='synthetic_data_generation'),
    path('api/synthetic_data_extended', handle_synthetic_data_extended, name='extended_synthetic_data_generation'),

]
