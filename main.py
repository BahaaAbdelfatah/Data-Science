import pickle
import matplotlib.pyplot as plt
import streamlit  as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

 
df = pd.read_csv('kc_house_data.csv')

st.title('House Price Prediction App')
st.info('Estimating the price of the houses in king country')


st.subheader("Data Preview")
st.write(df.head())
st.write(df.dtypes)
df.dtypes.value_counts().plot.pie()
# Display the plot in Streamlit
st.pyplot(plt)

st.subheader("Data Visualization")

plt.figure(figsize=(10, 7))
sns.histplot(df['price'], bins=30, kde=True)
plt.title('Price Distribution')
plt.xlabel('Price')
plt.ylabel('Frequency')

# Display the plot in Streamlit
st.pyplot(plt)

t = 2.5*10**6
df= df[df['price']<= t]

plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='floors', palette='viridis')
plt.title('Count of Houses by Number of Floors')
plt.xlabel('Number of Floors')
plt.ylabel('Count')
plt.xticks(rotation=0)  
plt.show()
# Display the plot in Streamlit
st.pyplot(plt)

plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='bedrooms', palette='viridis')
plt.title('Count of Houses by Number of Bedrooms')
plt.xlabel('Number of Bedrooms')
plt.ylabel('Count')
plt.xticks(rotation=0)  
plt.show()
st.pyplot(plt)

sns.scatterplot(data=df, x='sqft_living', y='price')
plt.title('Price vs. Square Footage of Living Space')
plt.xlabel('Square Footage (sqft_living)')
plt.ylabel('Price')
plt.show()
st.pyplot(plt)

sns.boxplot(data=df, x='bedrooms', y='price')
plt.title('Price Distribution by Number of Bedrooms')
plt.xlabel('Bedrooms')
plt.ylabel('Price')
plt.show()
st.pyplot(plt)

plt.figure()
df.plot(kind = 'scatter', x = 'long', y = 'lat', alpha = 0.8, c = 'price',cmap=plt.get_cmap('jet'), figsize = (12,8))
plt.legend(['0.5', '1.0', '1.5', '2', '2.5', '3', '3.5', '4'])
plt.show()
st.pyplot(plt)

st.subheader("Comparison between true value & predicted value)")
image_path = r"C:\Users\PCX\Desktop\comparison.PNG"  # Replace with your image file path

# Display the image
st.image(image_path, caption="Your Image Caption", use_column_width=True)



st.subheader("Model Deployment (Random Forest Regressor)")
floors=st.text_input('floors')
bedrooms=st.text_input('bedrooms')
bathrooms=st.text_input('bathrooms')
sqft_living=st.text_input('sqft_living')
waterfront=st.text_input('waterfront')
view=st.text_input('view')
condition=st.text_input('condition')
grade=st.text_input('grade')
sqft_above=st.text_input('sqft_above')
sqft_basement=st.text_input('sqft_basement')
lat=st.text_input('lat')
long=st.text_input('long')
sqft_lote=st.text_input('sqft_lote')

st.sidebar.header('Feature Selection')
df=pd.DataFrame({'bedrooms':[bedrooms], 'bathrooms':[bathrooms], 'sqft_living':[sqft_living], 'waterfront':[waterfront],
              'view':[view], 'condition':[condition], 'grade':[grade], 'sqft_above':[sqft_above], 'sqft_basement':[sqft_basement],
              'lat':[lat], 'long':[long], 'sqft_lote':[sqft_lote], 'floors':[floors]}, index= [0])
con=st.sidebar.button('Confirm')





