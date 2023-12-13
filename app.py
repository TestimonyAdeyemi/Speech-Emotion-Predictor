import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt




def remove_duplicates(df):
    # Remove duplicates from the entire dataset
    df = df.drop_duplicates()

    # Display the cleaned dataset
    # st.write("### Dataset After Removing Duplicates:")
    # st.dataframe(df_cleaned)

    # Display column names of the cleaned dataset
    # st.write("### Column Names After Removing Duplicates:")
    # st.write(df_cleaned.columns.tolist())

    # Return the cleaned dataset for further use
    return df

def load_data():
    st.title("Dataset Uploader, Cleaner, and Analyzer")

    # Upload CSV file
    st.subheader("Step 1: Upload your data")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        # Read the dataset
        df = pd.read_csv(uploaded_file)

        # Display the original dataset
        st.write("Your Dataset:")
        st.dataframe(df.head(5))

        # Option to clean the dataset
        st.subheader("Step  2: Remove Duplicates")
        if st.button("Remove Duplicates"):
            # Remove duplicates when the button is clicked
            df = remove_duplicates(df)
            st.success("Duplicates removed")

load_data()

st.subheader("Step 3: Aanlyze Data")
if st.button("Aanlyze Data"):
    # Descriptive Statistics
    st.write("#### Descriptive Statistics:")
    st.write(df.describe())

    # Correlation Heatmap
    st.write("#### Correlation Heatmap:")
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    st.pyplot()


