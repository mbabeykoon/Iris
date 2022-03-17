#import requred libraries
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB



st.write("""
# Simple Iris Flower Prediction App
This app predicts the **Iris flower** type!
""")

st.sidebar.header('User Input Parameters')
#get data from user
def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

iris = datasets.load_iris()
X = iris.data
Y = iris.target

#function to predict class 
def pred(classi):
    clf= classi()
    clf.fit(X, Y)
    prediction = clf.predict(df)
    prediction_proba = clf.predict_proba(df)
    return prediction


#function to assign predicted class into data frame
def prediction_df():
    pred1=pred(RandomForestClassifier)
    pred2 = pred(LogisticRegression)
    pred3=pred(DecisionTreeClassifier)
    pred4 = pred(KNeighborsClassifier)
    pred5=pred(LinearDiscriminantAnalysis)
    pred6 = pred(GaussianNB)
    
    
    predict = {'RandomForestClassifier': iris.target_names[pred1],
            'LogisticRegression': iris.target_names[pred2],
            'DecisionTreeClassifier': iris.target_names[pred3],
            'KNeighborsClassifier': iris.target_names[pred4],
            'LinearDiscriminantAnalysis': iris.target_names[pred5],
            'GaussianNB': iris.target_names[pred6]
            }
    predicted = pd.DataFrame(predict, index=[0])
    return predicted


predicted_df = prediction_df()


st.subheader('Class labels and their corresponding index number')
st.write(iris.target_names)

st.subheader('Prediction from different classifiers')
st.write(predicted_df)