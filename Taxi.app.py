import streamlit as st
import pandas as pd
import joblib
import mysql.connector
import plotly.express as px
import matplotlib.pyplot as plt
import requests
import io
import urllib.error

def establecer_conexion_mysql():
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="gmbb1245",
            database="proyecto_ny"
        )
        return conn
    except Exception as e:
        st.error(f"Error al conectar a la base de datos: {e}")
        return None

conexion_mysql = establecer_conexion_mysql()  # Establece la conexión a la base de datos

model_url = 'https://github.com/marcebalzarelli/Modelo_Prediccion_cantidad_de_taxis/blob/main/modelo_random_forest.joblib?raw=true'

# Descargar el modelo desde GitHub
response = requests.get(model_url)
model_data = io.BytesIO(response.content)

# Cargar el modelo
loaded_model = joblib.load(model_data)

promedio_pasajeros = 147049.8586065574 # Promedio de pasajeros

def predict_trips(avg_temperature, selected_dia_semana, tipo_taxi):
    dias_semana = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado']
    dia_semana_encoded = [0] * 6
    dia_semana_encoded[dias_semana.index(selected_dia_semana)] = 1
    tipo_taxi_encoded = 1 if tipo_taxi == 'yellow' else 0


    input_data = {
    'Pasajeros_por_dia': [promedio_pasajeros],
    'avg_temperature': [avg_temperature],
    'Tipo_de_Taxi_green': [0],  
    'Tipo_de_Taxi_yellow': [1], 
    'DiaSemana_Jueves': [0],  
    'DiaSemana_Lunes': [0],  
    'DiaSemana_Martes': [0],  
    'DiaSemana_Miércoles': [1],  
    'DiaSemana_Sábado': [0], 
    'DiaSemana_Viernes': [0] 
    }


    input_df = pd.DataFrame(input_data)

    prediction = loaded_model.predict(input_df)# Realizo predicción

    return prediction

st.write('<h1 style="text-align:center;">Analytica Data Solutions</h1>', unsafe_allow_html=True)
st.write('<h2 style="text-align:center;">Transformando Datos en Decisiones</h2>', unsafe_allow_html=True)
st.title('Predicción de Cantidad de Viajes de Taxis por día')# Configuro Streamlit
st.write('Esta aplicación predice la cantidad de viajes en taxi por día en la ciudad de Nueva York.')

avg_temperature = st.slider('Temperatura Promedio del día (°C)', min_value=-10.0, max_value=40.0, step=0.1, value=20.0)# Entrada de datos del usuario
dia_semana = st.selectbox('Día de la Semana', ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado'])
tipo_taxi = st.selectbox('Tipo de Taxi', ['green', 'yellow'])

if st.button('Realizar Predicción'):# Realizo predicción
    prediction = predict_trips(avg_temperature, dia_semana, tipo_taxi)
    st.write(f'La cantidad estimada de viajes en taxi es: {prediction[0]:.2f}')
       
    temperatura_ingresada = avg_temperature # Obtengo los valores ingresados por el usuario
    dia_semana_ingresado = dia_semana
    tipo_taxi_ingresado = tipo_taxi
    pasajeros_por_dia_ingresado = promedio_pasajeros

    if conexion_mysql is not None: # Inserto los datos en la tabla_combinada de MySQL
        cursor = conexion_mysql.cursor()
        try:
            # Consulta SQL
            consulta = "INSERT INTO tabla_combinada (Pasajeros_por_dia, Viajes_por_dia, Tipo_de_Taxi, DiaSemana, avg_temperature) VALUES (%s, %s, %s, %s, %s)"
            # Convierto a tipos de datos compatibles con MySQL
            valores = (float(pasajeros_por_dia_ingresado), float(prediction[0]), tipo_taxi_ingresado, dia_semana_ingresado, float(temperatura_ingresada))
            
            cursor.execute(consulta, valores)
            conexion_mysql.commit()
            st.success("Datos ingresados en la base de datos correctamente.")
        except Exception as e:
            st.error(f"Error al insertar datos en la base de datos: {e}")
        finally:
            cursor.close()
   
# Información adicional
st.write('Este modelo utiliza Random Forest para hacer predicciones. Los datos de entrada incluyen la temperatura promedio, el día de la semana y el tipo de taxi. Los datos ingresados serán guardados en nuestra base de datos para continuar entrenando el modelo.')

st.title('SonorAirAPI')

def calidad_aire_2020(ubicacion):
    if conexion_mysql is None:
        return {'error': 'Error al conectar a la base de datos'}

    try:
        # Creamos un cursor para ejecutar consultas SQL
        cursor = conexion_mysql.cursor(dictionary=True)
        
        # Consultamos SQL para obtener los datos de calidad del aire
        sql_query = """
        SELECT Name, Measure, Data_Value
        FROM calidad_del_aire
        WHERE Geo_Place_Name = %s AND Start_Date LIKE %s
        """
        
        # Convertimos la ubicación proporcionada a minúsculas
        ubicacion = ubicacion.lower()
        
        # Ejecutamos la consulta SQL
        cursor.execute(sql_query, (ubicacion, '2020%'))
        
        # Obtenemos los resultados
        calidad_aire_data = cursor.fetchall()
        
        # Verificamos si se encontraron datos
        if not calidad_aire_data:
            return {'error': 'No se encontraron datos para la ubicación'}
        
        # Creamos un diccionario de contaminantes y valores correspondientes
        contaminantes = ['Particulate Matter (PM2.5)', 'Particulate Matter (PM10)', 'Ozone (O3)', 'Nitrogen Dioxide (NO2)', 'Sulfur Dioxide (SO2)']
        calidad_aire_dict = {}
        
        for contaminante in contaminantes:
            calidad_aire_dict[contaminante] = [row for row in calidad_aire_data if row['Name'] == contaminante]

        return {'ubicacion': ubicacion, 'anio': '2020', 'datos_calidad_aire': calidad_aire_dict}
    
    except Exception as e:
        return {'error': f'Error al obtener datos de calidad del aire'}
    
def contaminacion_sonora(borough):
    if conexion_mysql is None:
        return {'error': 'Error al conectar a la base de datos'}

    try:
        # Creamos un cursor para ejecutar consultas SQL
        cursor = conexion_mysql.cursor(dictionary=True)
        
        # Consultamos SQL para obtener los datos de contaminación sonora
        sql_query = """
        SELECT YEAR(fecha) AS anio, SUM(engine_sounds) AS total_engine_sounds,
               SUM(alert_signal_sounds) AS total_alert_signal_sounds,
               SUM(total_sounds) AS total_sounds, MAX(borough_name) AS borough_name
        FROM conta_sonora
        WHERE LOWER(borough_name) LIKE %s AND YEAR(fecha) IN (2017, 2018, 2019)
        GROUP BY YEAR(fecha)
        """
    
        parametro_borough = f"%{borough.lower()}%"
        
        # Ejecutamos la consulta SQL
        cursor.execute(sql_query, (parametro_borough,))
        
        # Obtenemos los resultados
        contaminacion_sonora_data = cursor.fetchall()

        newBorough = contaminacion_sonora_data[0]["borough_name"].strip()
        
        # Verificamos si se encontraron datos
        if not contaminacion_sonora_data:
            return {'error': f'No se encontraron datos para el municipio: {borough}'}
        
        # Creamos un diccionario de estadísticas de contaminación sonora por año
        estadisticas_contaminacion = {}
        for fila in contaminacion_sonora_data:
            anio = fila['anio']
            estadisticas_contaminacion[anio] = {
                'total_engine_sounds': int(fila['total_engine_sounds']),
                'total_alert_signal_sounds': int(fila['total_alert_signal_sounds']),
                'total_sounds': int(fila['total_sounds'])
            }

        return {'ubicacion': newBorough, 'estadisticas_contaminacion': estadisticas_contaminacion}
    
    except Exception as e:
        return {'error': f'Error al obtener datos de contaminación sonora'}
    
# Ahora puedes definir la variable 'funcion'
funcion = st.selectbox('Selecciona una función', ('Calidad del Aire 2020', 'Contaminación Sonora'))

# Si el usuario selecciona "Calidad del Aire 2020"
if funcion == 'Calidad del Aire 2020':
    ubicacion = st.text_input('Ingrese la ubicación:')
    if st.button('Consultar'):
        resultado = calidad_aire_2020(ubicacion)
        st.write(resultado)

# Si el usuario selecciona "Contaminación Sonora"
elif funcion == 'Contaminación Sonora':
    borough = st.text_input('Ingrese el municipio:')
    if st.button('Consultar'):
        resultado = contaminacion_sonora(borough)
        st.write(resultado)

url_csv = 'https://drive.google.com/uc?id=12OIQyeSJeBjJhEiWR0VcChuIds3bGzhe'# Cargo CSV

try:
    df = pd.read_csv(url_csv)
except urllib.error.HTTPError as e:
    print(f"Error al cargar los datos: {e}")

viajes_por_dia_semana = df.groupby('DiaSemana')['Viajes_por_dia'].sum().reset_index()# Agrupo los datos por día de la semana y sumo los viajes

col1, col2 = st.columns(2)# Creo dos columnas en Streamlit

# Gráfico de barras para la cantidad de viajes por día de la semana
with col1: 
    st.title('Cantidad de Viajes por Día de la Semana')
    fig_bar = px.bar(viajes_por_dia_semana, x='DiaSemana', y='Viajes_por_dia', color='DiaSemana',
                     labels={'DiaSemana': 'Día de la Semana', 'Viajes_por_dia': 'Cantidad de Viajes'})
    st.plotly_chart(fig_bar, use_container_width=True)

# Boxplot para visualizar la distribución de viajes por día
with col1:
    st.title('Distribución de Viajes por Día')
    fig_box = px.box(df, x='DiaSemana', y='Viajes_por_dia', color='DiaSemana')
    st.plotly_chart(fig_box, use_container_width=True)

# Histograma para la distribución de la temperatura promedio
with col2:
    st.title('Distribución de la Temperatura Promedio')
    fig_hist = px.histogram(df, x='avg_temperature', nbins=30)
    st.plotly_chart(fig_hist, use_container_width=True)

# Gráfico de dispersión para la relación entre temperatura y viajes por día
with col2:
    st.title('Temperatura vs. Cantidad de Viajes por Día')
    fig_scatter = px.scatter(df, x='avg_temperature', y='Viajes_por_dia', color='DiaSemana',
                             labels={'avg_temperature': 'Temperatura Promedio', 'Viajes_por_dia': 'Cantidad de Viajes'})
    st.plotly_chart(fig_scatter, use_container_width=True)


st.write('Hecho por [Marcela Balzarelli, Pablo Barchiesi, Jorgelina Ramos, Michael Martinez]')# Texto de pie

##if conexion_mysql is not None:
  ##  conexion_mysql.close()




