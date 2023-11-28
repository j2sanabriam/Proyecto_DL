import streamlit as st
from PIL import Image
# import auxiliar as aux
# import shap


st.set_page_config(page_title="TechScan NN", layout="wide")
# st.title("TechScan NN \n Reconocimiento de Texto en Imágenes de Equipos Eléctricos")
st.title("TechScan NN")
st.title("Reconocimiento de Texto en Imágenes de Equipos Eléctricos")
st.write("Esta aplicación permite clasificar imágenes en 4 clases diferentes (trasformadores, placas de transformadores, características de postros y placas de postes). Adicionalmente, si la imagen pertence a las clases de placas, se ejecutan modelos de detección de objetos y aplicación de máscaras para reconocer el texto. Se realizar el procesamiento una imagen a la vez.")

if 'file' not in st.session_state:
    st.session_state['file'] = list()
    st.session_state['model'] = None

if not st.session_state['file']:
    # st.session_state['model'] = aux.load_model()

    # cargar archivo CSV
    # uploaded_file = st.file_uploader("Carga un Archivo", type=['csv'], accept_multiple_files=False)
    uploaded_file = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png"], accept_multiple_files=False)

    # Acciones al ejecutar botón Realizar Predicción
    if st.button("Realizar Reconocimiento") and uploaded_file is not None:

        st.subheader("Imagen Original")
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)






        # Cargar datos del archivo CSV
        # st.subheader("Datos Cargados")
        # df = aux.read_file(uploaded_file)
        # st.write(df)
        # st.session_state['file'].append(df)

        # Limpieza y transformación de los datos
        # st.subheader("Datos Luego de Limpieza y Trasformación")
        # df_p = aux.transform(df)


        # Predicción
        # st.subheader("Datos con Predicción")
        # y_pred = st.session_state['model'].predict(df_p.values)
        # df['prediccion_incurrira_mora'] = y_pred
        # df['prediccion_incurrira_mora'] = df['prediccion_incurrira_mora'].replace([0, 1], ['No', 'Si'])

        # Porcentajes de predicción
        # y_pred_proba = st.session_state['model'].predict_proba(df_p.values)
        # df = aux.add_numpy_to_dataframe(y_pred_proba, df)
        # st.write(df)

        # Gráfica importancia Features de entrada
        # column_headers = list(df_p.columns.values)
        # explainer = shap.Explainer(st.session_state['model'], df_p.values)
        # shap_values = explainer(df_p.values)
        # shap.summary_plot(shap_values, plot_type='violin', feature_names=column_headers)

        # Descargar archivo con predicción
        # csv = df.to_csv(index=False, sep=";").encode('utf-8')
        # filename = uploaded_file.name.split('.')[0] + "_predict.csv"
        # if st.download_button(
            # "Descargar Predicción",
            # csv,
            # filename,
            # "text/csv",
            # key='download-csv'
        # ):
            # st.write("Archivo Descargando")

else:
    if st.button("Reiniciar"):
        st.session_state['file'] = list()
        st.experimental_rerun()