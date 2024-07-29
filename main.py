import pandas as pd 
import numpy as np 
import pickle 
import streamlit as st 
from PIL import Image 
import seaborn as sns
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
import time
from sklearn.metrics import mean_absolute_error

csv1 = r"D:\aws_datascience\Training set.csv"
csv2 = r"D:\aws_datascience\Test set.csv"
df1 = pd.read_csv(csv1)
df2 = pd.read_csv(csv2)
df = pd.concat([df1, df2])

df = df[['Height','Sex','Weight']]
df = df.rename(columns={'Height':'height','Sex':'sex','Weight':'weight'})
df = df[(df['height']<500)&(df['weight']<150)&(df['height']>100)]


# loading in the model to predict on the data 
pickle_in = open('wt_pred_xgb_model.pkl', 'rb') 
model = pickle.load(pickle_in) 

with open('encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

def welcome(): 
	return 'welcome all'

# defining the function which will make the prediction using 
# the data which the user inputs 
def wt_pred(sex, height):
    new_data1 = {'sex': [sex], 'height': [height]}
    new_df1 = pd.DataFrame(new_data1, index=[0])
	# Handle categorical variables
    categorical_cols = ['sex']
    numerical_cols = ['height']
    new_data_encoded1 = encoder.transform(new_df1[categorical_cols])
    new_df_encoded1 = pd.concat([new_df1[numerical_cols], pd.DataFrame(new_data_encoded1, columns=encoder.get_feature_names_out())], axis=1)
    predicted_weight = model.predict(new_df_encoded1) 
    return predicted_weight

def DataViz():
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    sns.scatterplot(data=df, x='height', y='weight', hue='sex', palette=['blue', 'red'], alpha= 0.2, ax=axes[0,0])
    axes[0,0].set_title('The plot between weight and heighht')
    sns.histplot(data=df, x='height', ax=axes[0,1], hue='sex')
    axes[0,1].set_title('Histogram of height for both merged')
    for f in df.select_dtypes(exclude=np.number):
        sns.boxplot(x=f, y='weight', data=df, ax=axes[1,0])
    sns.histplot(data=df, x='weight', ax=axes[1,1], hue='sex')
    axes[1,1].set_title('Histogram of weight for both merged')
    plt.show()


# this is the main function in which we define our webpage 
def main(): 
	# giving the webpage a title 
	# st.title("Weight Prediction Demo") 
	
	# here we define some of the front end elements of the web page like 
	# the font and background color, the padding and the text to be displayed 
	html_temp = """ 
	<div style ="background-color:yellow;padding:13px"> 
	<h1 style ="color:black;text-align:center;">Weight Prediction App</h1> 
	</div> 
	"""
	
	# this line allows us to display the front end aspects we have 
	# defined in the above code 
	st.markdown(html_temp, unsafe_allow_html = True) 
	
	# the following lines create text boxes in which the user can enter 
	# the data required to make the prediction
	# Create the dropdown
	options = ['Male', 'Female']
	sex = st.selectbox('Select gender:', options) 
	height = st.number_input(label="enter the height in cm (in integer format)", min_value=100) 
	result = 0 
	
	# the below line ensures that when the button called 'Predict' is clicked, 
	# the prediction function defined above is called to make the prediction 
	# and store it in the variable result 
	if st.button("Predict the weight "): 
		result = wt_pred(sex,height)[0]
	st.success(f'The weight is {round(float(result),2)} kg') 

	if st.button("Learn more About the training data"): 
		DataViz() 
		st.pyplot(plt)
		st.dataframe(df.groupby('sex').agg({'height':[min, max, np.mean, np.median], 'weight':[min, max, np.mean, np.median]}).reset_index())
	
if __name__=='__main__': 
	main()
