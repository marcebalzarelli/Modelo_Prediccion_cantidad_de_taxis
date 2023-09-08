import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import matplotlib.pyplot as plt
import requests
import io
import urllib.error


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

    prediction = loaded_model.predict(input_df)  # Realizo predicción

    return prediction

st.write('<h1 style="text-align:center;">Analytica Data Solutions</h1>', unsafe_allow_html=True)
st.write('<h2 style="text-align:center;">Transformando Datos en Decisiones</h2>', unsafe_allow_html=True)
st.title('Predicción de Cantidad de Viajes de Taxis por día')  # Configuro Streamlit
st.write('Esta aplicación predice la cantidad de viajes en taxi por día en la ciudad de Nueva York.')

avg_temperature = st.slider('Temperatura Promedio del día (°C)', min_value=-10.0, max_value=40.0, step=0.1, value=20.0)  # Entrada de datos del usuario
dia_semana = st.selectbox('Día de la Semana', ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado'])
tipo_taxi = st.selectbox('Tipo de Taxi', ['green', 'yellow'])

if st.button('Realizar Predicción'):  # Realizo predicción
    prediction = predict_trips(avg_temperature, dia_semana, tipo_taxi)
    st.write(f'La cantidad estimada de viajes en taxi es: {prediction[0]:.2f}')
  
# Información adicional
st.write('Este modelo utiliza Random Forest para hacer predicciones. Los datos de entrada incluyen la temperatura promedio, el día de la semana y el tipo de taxi. Los datos ingresados serán guardados en nuestra base de datos para continuar entrenando el modelo.')

# Cargo CSV
url_csv1 = 'https://drive.google.com/uc?id=1VaDQnnVOlJ3afBWq5URA02ZdNRTFLDYT'#Calidad del aire
url_csv2 = 'https://drive.google.com/uc?id=16WANdrNvUXY3rJIlDDTRFfmRvnnx0gmc'#Cotaminación sonora

@st.cache_data
def load_data(url):
    data = pd.read_csv(url)
    return data


df1 = load_data(url_csv1)
df2 = load_data(url_csv2)

df2['fecha'] = pd.to_datetime(df2['fecha'])


st.title('SonorAirAPI')

# Función para obtener datos de calidad del aire
def calidad_aire_2020(ubicacion):
    try:
        
        filtered_data = df1[(df1['Start_Date'].str.startswith('2020')) &# Filtro los datos
                            (df1['Geo Place Name'].str.lower() == ubicacion.lower())]
        
        if filtered_data.empty:# Verifico si se encontraron datos
            return {'error': 'No se encontraron datos para la ubicación'}
    
        calidad_aire_dict = {} # Creo un diccionario con los datos de calidad del aire
        for contaminante in filtered_data['Name'].unique():
            contaminante_data = filtered_data[filtered_data['Name'] == contaminante]
            calidad_aire_dict[contaminante] = {
                'Measure': contaminante_data['Measure'].tolist(),
                'Data Value': contaminante_data['Data Value'].tolist(),
            }

        return {'ubicacion': ubicacion, 'anio': '2020', 'datos_calidad_aire': calidad_aire_dict}

    except Exception as e:
        return {'error': f'Error al obtener datos de calidad del aire: {e}'}

# Función para obtener datos de contaminación sonora
def contaminacion_sonora(borough):
    try:
        
        filtered_data = df2[(df2['borough_name'].str.lower().str.contains(borough.lower())) &#Filtro los datos
                            (df2['fecha'].dt.year.isin([2017, 2018, 2019]))]

        if filtered_data.empty:# Verifico si se encontraron datos
            return {'error': f'No se encontraron datos para el municipio: {borough}'}

        estadisticas_contaminacion = {} # Calculo estadísticas de contaminación sonora por año
        for year in [2017, 2018, 2019]:
            year_data = filtered_data[filtered_data['fecha'].dt.year == year]
            estadisticas_contaminacion[year] = {
                'total_engine_sounds': year_data['engine_sounds'].sum(),
                'total_alert_signal_sounds': year_data['alert_signal_sounds'].sum(),
                'total_sounds': year_data['total_sounds'].sum(),
            }

        return {'ubicacion': borough, 'estadisticas_contaminacion': estadisticas_contaminacion}

    except Exception as e:
        return {'error': f'Error al obtener datos de contaminación sonora: {e}'}

    
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

url_csv = 'https://drive.google.com/uc?id=1Gp2DZoJYhfNnvTa5xM1NKz5R31Fr0Xmr'# Cargo CSV

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





