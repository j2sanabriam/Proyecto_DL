import streamlit as st
import auxiliar as aux
import shap


st.set_page_config(page_title="SEMPLI App", layout="wide")
image_url = 'https://github.com/j2sanabriam/Proyecto_Final_ML/blob/main/App/img/sempli.png?raw=true'
st.image(image_url)
# st.image("img/sempli.png")
# st.title("Sempli App")
st.write("Esta aplicación permite predecir si clientes potenciales de Sempli incorrirá en mora, para así apoyar la decisión de otorgar o no tarjetas de crédito empresariales. Puede realizar la predicción para uno o varios cliente potenciales a través de un archivo en formato CSV.")

if 'file' not in st.session_state:
    st.session_state['file'] = list()
    st.session_state['model'] = None

if not st.session_state['file']:
    st.session_state['model'] = aux.load_model()

    # cargar archivo CSV
    uploaded_file = st.file_uploader("Carga un Archivo", type=['csv'], accept_multiple_files=False)

    # Acciones al ejecutar botón Realizar Predicción
    if st.button("Realizar Predicción") and uploaded_file is not None:

        # Cargar datos del archivo CSV
        st.subheader("Datos Cargados")
        df = aux.read_file(uploaded_file)
        st.write(df)
        st.session_state['file'].append(df)

        # Limpieza y transformación de los datos
        # st.subheader("Datos Luego de Limpieza y Trasformación")
        df_p = aux.transform(df)
        # st.write(df_p)

        # Predicción
        st.subheader("Datos con Predicción")
        y_pred = st.session_state['model'].predict(df_p.values)
        df['prediccion_incurrira_mora'] = y_pred
        df['prediccion_incurrira_mora'] = df['prediccion_incurrira_mora'].replace([0, 1], ['No', 'Si'])

        # Porcentajes de predicción
        y_pred_proba = st.session_state['model'].predict_proba(df_p.values)
        df = aux.add_numpy_to_dataframe(y_pred_proba, df)
        st.write(df)

        # Gráfica importancia Features de entrada
        column_headers = list(df_p.columns.values)
        explainer = shap.Explainer(st.session_state['model'], df_p.values)
        shap_values = explainer(df_p.values)
        shap.summary_plot(shap_values, plot_type='violin', feature_names=column_headers)

        # Descargar archivo con predicción
        csv = df.to_csv(index=False, sep=";").encode('utf-8')
        filename = uploaded_file.name.split('.')[0] + "_predict.csv"
        if st.download_button(
            "Descargar Predicción",
            csv,
            filename,
            "text/csv",
            key='download-csv'
        ):
            st.write("Archivo Descargando")

else:
    if st.button("Reiniciar"):
        st.session_state['file'] = list()
        st.experimental_rerun()