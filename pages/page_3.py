import streamlit as st
import pandas as pd

df  = pd.read_pickle('./data/results_2021.pickle')
st.bar_chart(df.index.unique()[:10])
st.dataframe(df)