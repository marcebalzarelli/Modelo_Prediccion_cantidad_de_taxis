import streamlit as st
import pandas as pd
import joblib
import mysql.connector
import plotly.express as px
import matplotlib.pyplot as plt

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

model_filename = r'E:\000-USUARIOS\Pablo\Documentos\Documentos\Marce\Data y Machine Learning\HENRY\Proy. Final\Modelo\modelo_random_forest.joblib'# Cargo el modelo entrenado
loaded_model = joblib.load(model_filename)

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

# Cargo el archivo
archivo_csv = r'E:\000-USUARIOS\Pablo\Documentos\Documentos\Marce\Data y Machine Learning\HENRY\Proy. Final\Datasets_limpios\tabla_combinada.csv'  # Reemplaza 'tu_archivo.csv' con la ruta de tu archivo CSV
df = pd.read_csv(archivo_csv)

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

# Gráfico de barras apiladas para tipos de taxi "green" y "yellow"
st.title('Cantidad de Viajes por Tipo de Taxi')
fig_stacked_bar = px.bar(df, x='DiaSemana', y='Viajes_por_dia', color='Tipo_de_Taxi',
                         labels={'DiaSemana': 'Día de la Semana', 'Viajes_por_dia': 'Cantidad de Viajes'},
                         barmode='group')
st.plotly_chart(fig_stacked_bar, use_container_width=True)

st.write('Hecho por [Marcela Balzarelli, Pablo Barchiesi, Jorgelina Ramos, Michael Martinez]')# Texto de pie

##if conexion_mysql is not None:
  ##  conexion_mysql.close()




