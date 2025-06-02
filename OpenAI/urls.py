from django.urls import path
from .views import *

urlpatterns = [
    # path("api/", testing, name="testing"),
    # path(r'api/register', register, name='register'),
    # path(r'api/login', loginpage, name='login'),
    path('api/connect', connection, name='connection'),
    path("api/upload_only", upload_and_store_data, name="upload functionaliy only"),
    path("api/upload", upload_and_analyze_data, name='upload_data'),
    path('api/tableinfo', get_tableinfo, name='get_table_info'),
    path("api/tabledata", read_data, name='read_table_data'),
    path("api/read_db_table_data", read_db_table_data, name='read_table_data'),
    path("api/get_user_data", get_user_data, name='gwt_user_data'),
    path('api/delete_selected_tables', delete_selected_tables_by_name, name='delete_selected_tables_by_name'),
    path('api/delete_all_user_tables', delete_all_user_tables, name='delete_all_user_tables'),

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

    # Forecast with wyge
    # path('api/forecasts', forecast_sales, name='forecasting'),
    path('api/synthetic_data', handle_synthetic_data_api, name='synthetic_data_generation'),
    path('api/synthetic_data_extended', handle_synthetic_data_extended, name='extended_synthetic_data_generation'),

    # #Sql_agentic_system
    # path('api/process_files', processing_files, name='processing_files'),
    # path('api/query_making', query_system, name='querying'),

    # SAP
    path('api/hana_connect', hana_connection, name='connecting'),
    path('api/upload_data', upload_data, name='uploading'),
    path('api/hana_dataread', reading_data, name='hana_reading_data'),
    path('api/hana_delete', delete_table_api, name='hana_delete_data'),

    # Flespi
    path('api/download_flespi_data', download_flespi_data, name='download_flespi_data'),

    # Customised KPIS URLS
    path('api/kpi_process', get_prompt, name="kpi_process"),
    path('api/generate_code', kpi_code, name="kpi_code"),

    # Predefined KPI urls
    path('api/detect_type', getting_types, name="detecting_type"),
    path('api/predefined_kpi_process', predefined_kpi_getting, name="predefined_kpi_getting"),

    # models for prediction urls
    path(r'api/models', models, name='models'),
    path('api/model_predict', model_predict, name='model_predict'),

    # Dashboard
    path('api/dashboard', gen_plotly_response, name="plotly dashboard"),
    path('api/fill_missed_data', missing_data, name="missed_data_filling"),

    # #Sla Breach
    # path('api/sla_breach', sla_breach, name="plotly dashboard"),

    # #Payment Gateway
    # path("api/initiate-payment", initiate_payment, name="initiate_payment"),
    # path("api/payment-callback", payment_callback, name="payment_callback"),

    # Visualisation_api_updated
    path('api/ai_bot', gen_ai_bot, name="plotly_visualisation"),
    path('api/getting_column_description', col_description, name="getting_column_description"),

    # Hanabot
    path('api/process_doc', upload_and_process_file, name='hana_bot_api_process'),
    path('api/hana_querying', query_data, name='hana_bot_api_query'),

    # Datascout apis
    path('api/data_scout', create_data_with_data_scout, name='creating the data with datascout agent'),

    #Predictive Maintenence Apis:
    #path('api/sensor_data_upload', upload_sensor_data, name='uploading the sensor data_from the user.'),
    path('api/predictive_maintenence', predictive_maintenence, name='predcitive maintenence for the given data.'),

]
