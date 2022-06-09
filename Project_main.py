import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import model_selection
#from sklearn import datasets
#from sklearn.preprocessing import LabelEncoder

matplotlib.use('Agg')

st.title("UCI Breast Cancer Dataset Web Machine Learning App")

from PIL import Image

st.subheader("This is a Web App presentation of Analysis of above Data set")

image = Image.open("..\\data\\atomak.png")

st.image(image, use_column_width=True)


def main():
    activities = ['EDA', 'Visualization', 'Model', 'About Us']
    option = st.sidebar.selectbox('Selection Option: ', activities)

    #Build each of the ooptions
    #EDA Option
    if option == 'EDA':
        st.subheader('Exploratory Data Analysis')
        #Load dataset
        data = st.file_uploader('Upload dataset', type =['csv', 'xlsx', 'txt', 'json'])
        st.success('Data sucessfully loaded')

        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head(50))
        
            if st.checkbox('Display shape'):
                st.write(df.shape)

            if st.checkbox('Display columns'):
                st.write(df.columns)
        
            if st.checkbox('Select Multiple Columns'):
                selected_columns = st.multiselect('Select your preferred columns:', df.columns)
                df1 = df[selected_columns]
                st.dataframe(df1)
        
            if st.checkbox('Display Summary'):
                st.write(df.describe().T)

            if st.checkbox('Display Null Values'):
                st.write(df.isnull().sum())
            
            if st.checkbox('Display data types'):
                st.write(df.dtpes)
            
            if st.checkbox('Display correlation'):
                st.write(df.corr())

    #Visualization option

    elif option == 'Visualization':
        st.subheader('Visualization')
        #Load dataset
        data = st.file_uploader('Upload dataset', type =['csv', 'xlsx', 'txt', 'json'])
        st.success('Data sucessfully loaded')

        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head(50))

            if st.checkbox('Select multiple columns to plot'):
                selected_columns = st.multiselect('Select your preffered coluns', df.columns)
                df1 = df[selected_columns]
                st.dataframe(df1)

            if st.checkbox('Display Heatmap'):
                st.write(sns.heatmap(df.corr(), vmax=1, square=True, annot=True, cmap='viridis'))
                st.pyplot()
            
            if st.checkbox('Display Pairplot'):
                st.write(sns.pairplot(df, diag_kind='kde'))
                st.pyplot()
            
            if st.checkbox('Display Pie Chart'):
                all_columns = df.columns.to_list()
                pie_columns = st.selectbox('Select the columns to display', all_columns)
                pieChart = df[pie_columns].value_counts().plot.pie(autopct='%1.1f%%')
                st.write(pieChart)


    elif option == 'Model':
        pass

    elif option == 'About Us':
        pass


if __name__ == '__main__':
    main()

