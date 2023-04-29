import streamlit as st
from PIL import Image

st.title('おうまさん')
st.caption('これはおうまさんのテストアプリです')
# 画像
image = 'https://frame-illust.com/fi/wp-content/uploads/2017/03/9547.png'
st.image(image, width=200)