import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats
from scipy.stats import norm
import altair as alt
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
#import warnings
from sklearn.model_selection import cross_val_score, RepeatedKFold
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, BaggingRegressor, RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

st.set_page_config(
    page_title="YouTube Trend Analytics", page_icon="📊", initial_sidebar_state="expanded"
)

st.write(
"""
# 📊 YouTube Trend Analytics
"""
)


model = pickle.load(open('AssignmentPickle.pkl','rb'))

def pred2(X_test):
    Y_predict = model.predict(X_test)
    return(Y_predict)

def main():

    app_file = "application_data.csv"
    df_2 = pd.read_csv(app_file)

    st.write("""
    #### Our Predicted column is : No. of days a video stays trending
    #### Below are five features against which the PREDICTED COLUMN can be compared - 
    - No. of likes  
    - No. of dislikes 
    - No. of comments 
    - No. of views  
    - the hour at which the video was published
    """
    )

    features= ['published_hour','likes','view_count','comment_count','dislikes']

# selectbox for selecting which state to plot
    feature_selected = st.selectbox('Select feature to view', features, index=(0))

    result = pred2(df_2)
    df_2['result_pred'] = result

    import altair as alt
    chart = alt.Chart(df_2).mark_bar().encode(
                x=alt.X(feature_selected,),
                y=alt.Y('result_pred', title = 'No. of days the Video was trending', axis=alt.Axis(labelOverlap="greedy",grid=False)))
    
    st.altair_chart(chart, use_container_width=True)


if __name__=='__main__':
    main()





