import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score,mean_squared_error



@st.cache(allow_output_mutation=True)
def loadData():
    df = pd.read_csv('insurance.csv')
    return df

def pre_process(df):
    le = LabelEncoder() 
  
    df['sex']= le.fit_transform(df['sex']) 
    df['region']= le.fit_transform(df['region']) 
    df['smoker']= le.fit_transform(df['smoker']) 

    X = df.drop(['charges'], axis = 1)
    y = df.charges

    X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 0)

    return X_train, X_test, y_train, y_test, le

@st.cache(suppress_st_warning=True)
def lr(X_train, X_test, y_train, y_test):
    lg = LinearRegression()
    lg.fit(X_train, y_train)
    y_pred = lg.predict(X_test)
    score = r2_score(y_test, y_pred)

    return score, lg

def main():
	st.title("A Streamlit Demo! For checking accuracy of different Linear Regression Methods")
	data = loadData()
	X_train, X_test, y_train, y_test, le = pre_process(data)

	
	if st.checkbox('Show Raw Data'):
		st.subheader("Showing raw data>>>")	
		st.write(data.head())


	
	choose_model = st.sidebar.selectbox("Choose the ML Model",
		["NONE","Linear regression", "LR - Polynomial", "Ridge Regression"])

	if(choose_model == "Linear regression"):
		score, tree = lr(X_train, X_test, y_train, y_test)
		st.text("Accuracy of Simple Linear Regression model is: ")
		st.write(score * 100,"%")
		#st.text("Report of Decision Tree model is: ")
		#st.write(report)

if __name__ == "__main__":
	main()


