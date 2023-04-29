import streamlit as st
from PIL import Image
import datetime

st.title('おうまさん')
st.caption('これはおうまさんのテストアプリです')
st.subheader('自己紹介')
st.text('youtubeの「競馬予想で始めるデータ分析・機械学習」を参考に作成')

# コードをいい感じに表示
code = '''
import stream lit as st

st.titile
'''
st.code(code, language='python')

# 画像
image = 'https://frame-illust.com/fi/wp-content/uploads/2017/03/9547.png'
st.image(image, width=200)


# # テキストボックス（TB）
# # TBに値が入ってくるのは画面がリロードされたとき。TBからカーソルが外れると画面が自動でリロードされる。
# name = st.text_input('名前')
# adress = st.text_input('住所')
# print(name)

# # ボタン
# # 押されている：True、押されていない：False
# submit_btn = st.button('送信')
# cancel_btn = st.button('キャンセル')
# print(f'submit_btn: {submit_btn}')
# print(f'cancel_btn: {cancel_btn}')
# if submit_btn:
#     st.text(f'ようこそ！{name}さん！')

# 画面がいちいちリロードされない設定
with st.form(key='profile_form'):
    #テキスト
    name = st.text_input('名前')
    adress = st.text_input('住所')

    # セレクトボックス
    age_category = st.selectbox(
        '年齢層',
        ('子供', '大人')
    )
    # ラジオ
    sex_category = st.radio(
        '性',
        ('雄', '雌')
    )
    # 複数選択
    hobby = st.multiselect(
        '趣味',
        ('走ること', '草を食う', '交尾')
    )
    # 日付
    start_date = st.date_input(
        '開始日',
        datetime.date(2023, 4, 28)
    )

    # ボタン
    submit_btn = st.form_submit_button('送信')
    cancel_btn = st.form_submit_button('キャンセル')

    if submit_btn:
        st.text(f'ようこそ！{name}さん！{adress}に書籍を送りました！')
        st.text(f'年齢層：{age_category}、性：{sex_category}')
        st.text(f'趣味：{", ".join(hobby)}')

