import pandas as pd
import streamlit as st
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer, make_column_selector
import numpy as np
from scipy.sparse import csr_matrix
import requests
import streamlit.components.v1 as components
import shap



def load_original_data():
    data = pd.read_csv("./data/solicitudes_tc_dataset.csv", sep=";")
    data2 = data.drop(['moroso'], axis=1)
    return data2


def read_file(file):
    df = pd.read_csv(file, sep=";")
    return df


@st.cache_data
def load_model():
    """
    # path = 'https://github.com/j2sanabriam/Proyecto_Final_ML/blob/64ff64b0098be2349e4b6ea9a66e11fc923cdac3/App/models/SVM.pkl?raw=true'
    path = 'models/SVM.pkl'

    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model
    """
    path = 'models/SVM_2.pkl'

    url = 'https://github.com/j2sanabriam/Proyecto_Final_ML/blob/a8ebbd81380d55e40b1d894e06d4a73b5fae3349/App/models/SVM.pkl?raw=true'
    r = requests.get(url)

    with open(path, 'wb') as file:
        file.write(r.content)

    with open(path, 'rb') as f:
        model = pickle.load(f)

    return model



def eliminar_columnas_no_utiles(X):
    columnas_no_utiles = ['contract_number', 'card_type', 'business_dni_type', 'city', 'mora_ultimos_6meses',
                          'created_at']
    return X.drop(columnas_no_utiles, axis=1)


@st.cache_data
def create_pipeline():
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), make_column_selector(dtype_include=np.number)),
            ("cat", OneHotEncoder(), make_column_selector(dtype_include=object)),
        ]
    )
    pipeline = Pipeline(steps=[
        ('eliminar_columnas', FunctionTransformer(eliminar_columnas_no_utiles)),
        ('column_transformer', preprocessor)
    ])

    pipeline.fit(load_original_data())
    return pipeline



def transform (df):
    pipeline = create_pipeline()

    cat_names = pipeline['column_transformer'].transformers_[1][1].get_feature_names_out()
    num_names = pipeline['column_transformer'].transformers_[0][1].feature_names_in_
    col_names = list(num_names) + list(cat_names)

    result = pipeline.transform(df)
    datos = result.data
    indices = result.indices
    indptr = result.indptr

    csr_matrix_data = csr_matrix((datos, indices, indptr), shape=(1711, 106))
    df_p = pd.DataFrame(csr_matrix_data.toarray())
    df_p.columns = col_names

    return df_p

def add_numpy_to_dataframe(array , df):
    df2 = pd.DataFrame(array, columns=['probabilidad_No', 'probabilidad_Si'])
    df = pd.concat([df, df2], axis=1)
    return df


def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)