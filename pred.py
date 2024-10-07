import pandas as pd
import numpy as np
import streamlit  as st


# Replace 'model.joblib' with the path to your .joblib file
model = joblib.load('modelres.joblib')

# Now you can use the model
print(model)


st.title('House Price Prediction App')
st.info('Estimating the price of the houses in king country')


st.subheader("Model Deployment (Random Forest Regressor)")

# Collecting input from Streamlit
bedrooms = st.number_input('Bedrooms', min_value=0, value=0)
bathrooms = st.number_input('Bathrooms', min_value=0.0, value=0.0)
sqft_living = st.number_input('Square Footage (sqft_living)', min_value=0.0, value=0.0)
sqft_lot = st.number_input('Square Footage Lot (sqft_lot)', min_value=0.0, value=0.0)
floors = st.number_input('Floors', min_value=0, value=0)
waterfront = st.number_input('Waterfront (0 or 1)', min_value=0, max_value=1)
view = st.number_input('View (0-4)', min_value=0, max_value=4)
condition = st.number_input('Condition (1-5)', min_value=1, max_value=5)
grade = st.number_input('Grade (1-13)', min_value=1, max_value=13)
sqft_above = st.number_input('Square Footage Above', min_value=0.0, value=0.0)
sqft_basement = st.number_input('Square Footage Basement', min_value=0.0, value=0.0)


df_input = pd.DataFrame({
        'bedrooms': [int(bedrooms)],
        'bathrooms': [int(bathrooms)],
        'sqft_living': [float(sqft_living)],
        'sqft_lot': [float(sqft_lot)],
        'floors': [int(floors)],
        'waterfront': [int(waterfront)],
        'view': [int(view)],
        'condition': [int(condition)],
        'grade': [int(grade)],
        'sqft_above': [float(sqft_above)],
        'sqft_basement': [float(sqft_basement)]
 })

if st.sidebar.button('Confirm'):
        st.write(df_input)  # Display DataFrame for debugging
        result = model.predict(df_input)
        st.write(f"Predicted Price: {result[0]}")
   







