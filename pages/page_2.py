import streamlit as st
import datetime

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
