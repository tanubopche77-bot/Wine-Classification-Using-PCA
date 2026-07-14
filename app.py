import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.datasets import load_wine

st.set_page_config(
    page_title="Wine Classification Dashboard",
    page_icon="🍷",
    layout="wide"
)

model = joblib.load("wine_pipeline.pkl")

wine = load_wine()

feature_names = wine.feature_names
target_names = wine.target_names

df = pd.DataFrame(wine.data, columns=feature_names)
df["Target"] = wine.target

# ---------------- Sidebar ----------------

st.sidebar.title("🍷 Wine Dashboard")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🤖 Prediction",
        "📊 Dataset",
        "📈 PCA Analysis",
        "ℹ About"
    ]
)

# ---------------- Home Page ----------------

if page=="🏠 Home":

    st.markdown(
        "<h1 class='main-title'>Wine Classification Dashboard</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<p class='sub-title'>Machine Learning + PCA + Random Forest</p>",
        unsafe_allow_html=True
    )

    st.write("")

    c1,c2,c3,c4 = st.columns(4)

    with c1:
        st.metric("Dataset", "Wine")

    with c2:
        st.metric("Samples", wine.data.shape[0])

    with c3:
        st.metric("Features", wine.data.shape[1])

    with c4:
        st.metric("Classes", len(target_names))

    st.write("")

    left,right = st.columns([2,1])

    with left:

        st.subheader("📌 Project Overview")

        st.write("""
This project predicts **Wine Category**
using **Principal Component Analysis (PCA)**
and **Random Forest Classifier**.

### Technologies Used

- Python
- Streamlit
- PCA
- Random Forest
- Scikit-Learn
- Pandas
- NumPy
- Matplotlib
- Seaborn

### Machine Learning Workflow

✔ Data Loading

✔ StandardScaler

✔ PCA

✔ Model Training

✔ Prediction

✔ Visualization

✔ Model Comparison

✔ Pipeline

✔ Model Deployment

        """)

    with right:

        st.info("""
### Dashboard Features

🏠 Home

🤖 Prediction

📊 Dataset

📈 PCA Analysis

📉 Model Comparison

ℹ About Project
""")

    st.success("Dashboard Loaded Successfully ✅")
# ---------------- Prediction Page ----------------

elif page == "🤖 Prediction":

    st.title("🤖 Wine Prediction")

    st.write("Enter the wine feature values below and click Predict.")

    col1, col2 = st.columns(2)

    input_data = []

    with col1:
        alcohol = st.number_input("Alcohol", value=13.0)
        malic_acid = st.number_input("Malic Acid", value=2.0)
        ash = st.number_input("Ash", value=2.3)
        alcalinity = st.number_input("Alcalinity of Ash", value=19.0)
        magnesium = st.number_input("Magnesium", value=100.0)
        phenols = st.number_input("Total Phenols", value=2.5)
        flavanoids = st.number_input("Flavanoids", value=2.0)

    with col2:
        nonflav = st.number_input("Nonflavanoid Phenols", value=0.30)
        proanth = st.number_input("Proanthocyanins", value=1.6)
        color = st.number_input("Color Intensity", value=5.0)
        hue = st.number_input("Hue", value=1.0)
        od = st.number_input("OD280 / OD315", value=3.0)
        proline = st.number_input("Proline", value=1000.0)

    input_data = [[
        alcohol,
        malic_acid,
        ash,
        alcalinity,
        magnesium,
        phenols,
        flavanoids,
        nonflav,
        proanth,
        color,
        hue,
        od,
        proline
    ]]

    df_input = pd.DataFrame(
        input_data,
        columns=feature_names
    )

    if st.button("🔍 Predict Wine Class"):

        prediction = model.predict(df_input)[0]

        probability = model.predict_proba(df_input)

        st.success(
            f"✅ Predicted Wine Class : {target_names[prediction]}"
        )

        st.subheader("Prediction Probability")

        prob_df = pd.DataFrame({
            "Wine Class": target_names,
            "Probability (%)": np.round(
                probability[0] * 100,
                2
            )
        })

        st.dataframe(
            prob_df,
            use_container_width=True
        )

        st.bar_chart(
            prob_df.set_index("Wine Class")
        )

        st.download_button(
            label="📥 Download Prediction",
            data=prob_df.to_csv(index=False),
            file_name="wine_prediction.csv",
            mime="text/csv"
        )

# ---------------- Dataset Page ----------------

elif page == "📊 Dataset":

    st.title("📊 Wine Dataset Overview")

    df = pd.DataFrame(
        wine.data,
        columns=feature_names
    )

    df["Target"] = wine.target

    st.subheader("Dataset")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing Values", df.isnull().sum().sum())

    st.subheader("Dataset Statistics")

    st.dataframe(
        df.describe(),
        use_container_width=True
    )

    st.subheader("Target Distribution")

    target_count = df["Target"].value_counts()

    st.bar_chart(target_count)

# ---------------- PCA Analysis ----------------

elif page == "📈 PCA Analysis":

    st.title("📈 PCA Analysis")

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    x = wine.data
    y = wine.target

    scaler = StandardScaler()

    x_scaled = scaler.fit_transform(x)

    pca = PCA(n_components=7)

    x_pca = pca.fit_transform(x_scaled)

    pca_df = pd.DataFrame(
        x_pca,
        columns=[f"PC{i+1}" for i in range(7)]
    )

    pca_df["Target"] = y

    st.subheader("PCA Scatter Plot")

    fig, ax = plt.subplots(figsize=(8,6))

    sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        hue="Target",
        palette="Set1",
        ax=ax
    )

    st.pyplot(fig)

    st.subheader("Explained Variance")

    fig, ax = plt.subplots(figsize=(8,4))

    ax.plot(
        np.cumsum(
            pca.explained_variance_ratio_
        ),
        marker="o"
    )

    ax.set_xlabel("Principal Components")
    ax.set_ylabel("Cumulative Variance")

    st.pyplot(fig)

    st.subheader("Scree Plot")

    fig, ax = plt.subplots(figsize=(8,4))

    ax.bar(
        range(1,8),
        pca.explained_variance_
    )

    ax.set_xlabel("Principal Components")
    ax.set_ylabel("Eigen Values")

    st.pyplot(fig)

    st.subheader("Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(10,8))

    sns.heatmap(
        pd.DataFrame(
            wine.data,
            columns=feature_names
        ).corr(),
        cmap="coolwarm",
        annot=True,
        ax=ax
    )

    st.pyplot(fig)

# ---------------- About ----------------

elif page == "ℹ About":

    st.title("ℹ About Project")

    st.markdown("""
## 🍷 Wine Classification using PCA

This project demonstrates an end-to-end Machine Learning workflow.

### Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn

### Machine Learning Algorithms

- Logistic Regression
- Decision Tree
- Random Forest
- KNN
- SVM
- Naive Bayes

### PCA

Principal Component Analysis (PCA) is used to reduce dimensionality
while preserving maximum information.

### Model Deployment

The final model is deployed using Streamlit.

---
""")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Dataset", "Wine")

    with c2:
        st.metric("Features", "13")

    with c3:
        st.metric("Classes", "3")

    st.info(
        "Developed using Python, PCA and Random Forest."
    )

# ---------------- Footer ----------------

st.markdown("---")

st.markdown(
"""
<div style="text-align:center; color:gray;">

Made with ❤️ using Streamlit

<b>Wine Classification Dashboard</b>

</div>
""",
unsafe_allow_html=True
)
