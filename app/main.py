import plotly.graph_objects as go
import streamlit as st
import pickle as pickle
import pandas as pd
import numpy as np

def get_clean_data():
    data = pd.read_csv("data/data.csv")
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

    return data

def add_sidebar():
    st.sidebar.header("Cell Nuclei Measurements")
    data = get_clean_data()

    slider_labels = [
        ("Radius (Mean)", "radius_mean"),
        ("Texture (Mean)", "texture_mean"),
        ("Perimeter (Mean)", "perimeter_mean"),
        ("Area (Mean)", "area_mean"),
        ("Smoothness (Mean)", "smoothness_mean"),
        ("Compactness (Mean)", "compactness_mean"),
        ("Concavity (Mean)", "concavity_mean"),
        ("Concave Points (Mean)", "concave points_mean"),
        ("Symmetry (Mean)", "symmetry_mean"),
        ("Fractal Dimentsion (Mean)", "fractal_dimension_mean"),
        ("Radius (Standard Error)", "radius_se"),
        ("Texture (Standard Error)", "texture_se"),
        ("Perimeter (Standard Error)", "perimeter_se"),
        ("Area (Standard Error)", "area_se"),
        ("Smoothness (Standard Error)", "smoothness_se"),
        ("Compactness (Standard Error)", "compactness_se"),
        ("Concavity (Standard Error)", "concavity_se"),
        ("Concave Points (Standard Error)", "concave points_se"),
        ("Symmetry (Standard Error)", "symmetry_se"),
        ("Fractal Dimension (Standard Error)", "fractal_dimension_se"),
        ("Radius (Worst)", "radius_worst"),
        ("Texture (Worst)", "texture_worst"),
        ("Perimeter (Worst)", "perimeter_worst"),
        ("Area (Worst)", "area_worst"),
        ("Smoothness (Worst)", "smoothness_worst"),
        ("Compactness (Worst)", "compactness_worst"),
        ("Concavity (Worst)", "concavity_worst"),
        ("Concave Points (Worst)", "concave points_worst"),
        ("Symmetry (Worst)", "symmetry_worst"),
        ("Fractal Dimension (Worst)", "fractal_dimension_worst")
    ]

    input_dict = {}

    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label,
            min_value=float(0),
            max_value=float(data[key].max()),
            value=float(data[key].mean())
        )

    return input_dict

def get_scaled_value(input_dict):
    data = get_clean_data()
    X = data.drop(['diagnosis'], axis=1)

    scaled_dict = {}
    for key, value in input_dict.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value

    return scaled_dict

def get_radar_chart(input_data):
    input_data = get_scaled_value(input_data)
    categories = ['Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness', 'Compactness',
                  'Concavity', 'Concave Points', 'Symmetry', 'Fractal Dimension']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_mean'], input_data['texture_mean'],
            input_data['perimeter_mean'], input_data['area_mean'],
            input_data['smoothness_mean'], input_data['compactness_mean'],
            input_data['concavity_mean'], input_data['concave points_mean'],
            input_data['symmetry_mean'], input_data['fractal_dimension_mean']
        ],
        theta=categories,
        fill='toself',
        name='Mean Value'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_se'], input_data['texture_se'],
            input_data['perimeter_se'], input_data['area_se'],
            input_data['smoothness_se'], input_data['compactness_se'],
            input_data['concavity_se'], input_data['concave points_se'],
            input_data['symmetry_se'], input_data['fractal_dimension_se']
        ],
        theta=categories,
        fill='toself',
        name='Standard Error'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_worst'], input_data['texture_worst'],
            input_data['perimeter_worst'], input_data['area_worst'],
            input_data['smoothness_worst'], input_data['compactness_worst'],
            input_data['concavity_worst'], input_data['concave points_worst'],
            input_data['symmetry_worst'], input_data['fractal_dimension_worst']
        ],
        theta=categories,
        fill='toself',
        name='Worst Value'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        width=2050,
        height=550
    )

    return fig

def add_predictions(input_data):
    model = pickle.load(open("model/model.pkl", "rb"))
    scaler = pickle.load(open("model/scaler.pkl", "rb"))

    input_array = np.array(list(input_data.values())).reshape(1, -1)
    input_array_scaled = scaler.transform(input_array)

    prediction = model.predict(input_array_scaled)

    st.subheader("Cell Cluster Prediction")
    st.write("The cell cluster is: ")
    
    if prediction[0] == 0:
        st.write("<span class='diagnosis benign'>Benign</span>", unsafe_allow_html=True)
    else:
        st.write("<span class='diagnosis malignant'>Malignant</span>", unsafe_allow_html=True)

    st.write("Probability of being Benign: ", model.predict_proba(input_array_scaled)[0][0])
    st.write("Probability of being Malignant: ", model.predict_proba(input_array_scaled)[0][1])

    st.write("This app can assist medical professionals in making a diagnosis,"
             "but should not be used as a subsitute for a professional diagnosis.")


def main():
    st.set_page_config(
        page_title="Breast Cancer Predictor",
        page_icon=":female-doctor:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    with open("assets/style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

    input_data = add_sidebar()

    with st.container():
        st.title("Breast Cancer Predictor")
        st.write("This app predicts using a machine learning model whether a breast mass is"
                  "benign or malignant based on the measurements it receives from your tissue"
                  "sample. You can also update the measurements by hand using the sliders in"
                  "the sidebar.")
        st.write("This app is for educational purposes only.")

    col1, col2 = st.columns([4, 1])

    with col1:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)

    with col2:
        add_predictions(input_data)
        
if __name__ == '__main__':
    main()