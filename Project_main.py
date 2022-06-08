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
    if option == 'EDA':
        pass

    elif option == 'Visualization':
        pass

    elif option == 'Model':
        pass

    elif option == 'About Us':
        pass


if __name__ == '__main__':
    main()

