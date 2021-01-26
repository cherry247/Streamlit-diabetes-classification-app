import streamlit as st
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score, recall_score,classification_report

st.title("Diabetes Classifcation App")
st.markdown("Is the person suffering from diabetes? ")
st.sidebar.title('Diabetes Classification')

@st.cache(persist = True)
def load_data():
     data=pd.read_csv('diabetes.csv')
     target=data['Outcome']
     return data,target
def plot_correlation(data):
    corr=data.corr()
    fig,ax=plt.subplots()
    sns.heatmap(corr,cmap="Blues",annot=True,ax=ax)
    st.pyplot(fig)
    
def split(data):
    y=data['Outcome']
    x=data.drop('Outcome',axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)
    return x_train,x_test,y_train,y_test
#-----------load data----------------------#
data,target=load_data() #load csv file

#-----------plot raw data & summary-------------#
if st.sidebar.checkbox('Raw Data'): # display csv file
    st.write(data.head())
if st.sidebar.checkbox('Summary'): #display summary of the dataset
    st.write(data.describe())
    st.write(data['Outcome'].value_counts())
    plot_correlation(data)
#---------- train test split------------------#
x_train,x_test,y_train,y_test=split(data)
st.sidebar.subheader("Choose the classifier")
classifier=st.sidebar.selectbox("classifier",('Logistic regression','Guassian naive bayes','Decision tree'))
#------------print data info-------------------#
st.subheader("Outcome attribut distribution")
st.write("Original Diabetes True Values    : {0} ({1:0.2f}%)".format(len(data.loc[data['Outcome'] == 1]), (len(data.loc[data['Outcome'] == 1])/len(data.index)) * 100))
st.write("Original Diabetes false Values   : {0} ({1:0.2f}%)".format(len(data.loc[data['Outcome'] == 0]), (len(data.loc[data['Outcome'] == 0])/len(data.index)) * 100))
st.write("Training Diabetes True Values    : {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == 1]), (len(y_train[y_train[:] == 1])/len(y_train)) * 100))
st.write("Training Diabetes False Values   : {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == 0]), (len(y_train[y_train[:] == 0])/len(y_train)) * 100))
st.write("Test Diabetes True Values        : {0} ({1:0.2f}%)".format(len(y_test[y_test[:] == 1]), (len(y_test[y_test[:] == 1])/len(y_test)) * 100))
st.write("Test Diabetes False Values       : {0} ({1:0.2f}%)".format(len(y_test[y_test[:] == 0]), (len(y_test[y_test[:] == 0])/len(y_test)) * 100))

#-------------- logistic regression-------------------------------------------#
if classifier=='Logistic regression':
    st.sidebar.subheader("model hyperparameter")
    max_iter=st.sidebar.slider("maximum no.of iteration",100,400,key = 'max_iter')
    C=st.sidebar.number_input("regularisation",0.1,10.0,step=0.1,key = 'C_LR')

    if st.sidebar.button('classify',key='classify'):
        st.subheader("Logistic Regression Results")
        model=LogisticRegression(C=C,max_iter=max_iter)
        model.fit(x_train,y_train)
        accuracy=model.score(x_test,y_test)
        y_pred = model.predict(x_test)
        st.write("accuracy is",accuracy.round(2)*100)
        st.write("precision is",precision_score(y_test,y_pred,labels=['No diabetes','Diabetes']))
        st.write("recall is",recall_score(y_test,y_pred,labels=['No diabetes','Diabetes']))
#-------------------------- Decision tree ----------------------------------------------------#
if classifier=='Decision tree':
    if st.sidebar.button('classify'):
        model=DecisionTreeClassifier()
        model.fit(x_train,y_train)
        accuracy=model.score(x_test,y_test)
        y_pred = model.predict(x_test)
        st.write("accuracy is",accuracy.round(2)*100)
        st.write("precision is",precision_score(y_test,y_pred,labels=['No diabetes','Diabetes']))
        st.write("recall is",recall_score(y_test,y_pred,labels=['No diabetes','Diabetes']))
    
#-------------------------- Guassion NB ----------------------------------------------------#
if classifier=="Guassian naive bayes":
    if st.sidebar.button('classify'):
        model=GaussianNB()
        model.fit(x_train,y_train)
        accuracy=model.score(x_test,y_test)
        y_pred = model.predict(x_test)
        st.write("accuracy is",accuracy.round(2)*100)
        st.write("precision is",precision_score(y_test,y_pred,labels=['No diabetes','Diabetes']))
        st.write("recall is",recall_score(y_test,y_pred,labels=['No diabetes','Diabetes']))
       




