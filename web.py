import cv2
from tensorflow.keras.models import load_model
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import requests
from PIL import Image
from io import BytesIO

# 화면을 최대로 와이드 
st.set_page_config(layout="wide")

# 제목
st.write('# World Master')
st.write('# Prediction of handwritten English character')

# 모델 로드
@st.cache(allow_output_mutation=True)
def load():
    url = 'https://github.com/KwonBK0223/Handwriting_recognition_project_using_deep_learning/blob/main/maincnn.h5'
    r = requests.get(url)
    with open('maincnn.h5','wb') as f:
        f.write(r.content)        
    model = load_model('maincnn.h5')
    return model
model = load()

# 알파벳 대문자 레이블
labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

# 메인 예측 페이지
def home():
    CANVAS_SIZE = 192

    # 사용자로부터 숫자 입력 받기
    num_canvas = st.number_input('Enter the number of alphabets you want to enter(1~10)', min_value=1, max_value=10, value=2, step=1)

    # canvas 생성 및 예측 결과 계산
    predictions = ''

    # 5개 단위로 자르기 위해서 줄 나누기
    num_rows = num_canvas // 5 # 몫 => 줄 개수
    num_cols = num_canvas % 5  # 나머지 => 마지막줄
    for row in range(num_rows):
        col_list = st.columns(5)
        for i, col in enumerate(col_list):
            with col:
                canvas = st_canvas(
                    fill_color='#000000',
                    stroke_width=12,
                    stroke_color='#FFFFFF',
                    background_color='#000000',
                    width=CANVAS_SIZE,
                    height=CANVAS_SIZE,
                    drawing_mode='freedraw',
                    key=f'canvas{row}_{i}'  # row 값을 key에 추가
                )

                if canvas.image_data is not None:
                    img = canvas.image_data.astype(np.uint8)
                    img = cv2.resize(img, dsize=(28, 28))

                    x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    x = x.reshape((-1, 28, 28, 1))
                    y = model.predict(x).squeeze()

                    # 예측 결과의 최댓값 인덱스를 구함
                    pred_idx = np.argmax(y)
                    # 레이블에 해당하는 문자를 가져옴
                    pred_char = labels[pred_idx]

                    # 예측 결과 문자열에 추가
                    predictions += pred_char
                        
    if num_cols > 0:
        col_list = st.columns(num_cols)
        for i, col in enumerate(col_list):
            with col:
                canvas = st_canvas(
                    fill_color='#000000',
                    stroke_width=12,
                    stroke_color='#FFFFFF',
                    background_color='#000000',
                    width=CANVAS_SIZE,
                    height=CANVAS_SIZE,
                    drawing_mode='freedraw',
                    key=f'canvas{num_rows}_{i}'  # num_rows 값을 key에 추가
                )

                if canvas.image_data is not None:
                    img = canvas.image_data.astype(np.uint8)
                    img = cv2.resize(img, dsize=(28, 28))

                    x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    x = x.reshape((-1, 28, 28, 1))
                    y = model.predict(x).squeeze()

                    # 예측 결과의 최댓값 인덱스를 구함
                    pred_idx = np.argmax(y)
                    # 레이블에 해당하는 문자를 가져옴
                    pred_char = labels[pred_idx]

                    # 예측 결과 문자열에 추가
                    predictions += pred_char
    
    # 결과값 출력
    st.write('## Predictions : %s' % predictions)
 
# 개념설명 페이지
def page1():
    st.write("# What is CNN")
    st.write("제작중")
# 코드설명 페이지
def page2():
    st.write("# 모델링 결과")
    st.write("제작중")

# 팀원 페이지
def Team_Mate():
    url = 'https://github.com/KwonBK0223/Handwriting_recognition_project_using_deep_learning/blob/main/Image/PNU_Mark.png'
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    col1, col2 = st.columns([1,5])
    with col1:
        st.write("\n")
        st.write("\n")
        st.image(img, width = 200)
        st.write("\n")
        st.write("\n")
        st.write("\n")
        st.image(img, width = 200)
    with col2:
        st.write("# Leader")
        st.write("## Kwon Byeong Keun")
        st.write("#### PNU Matematics 17")
        st.write("#### house9895@naver.com")

        st.write("# Team mate")
        st.write("## Seong Da Som")
        st.write("#### PNU Mathematics 19")
        st.write("#### som0608@naver.com")
# 메뉴 생성
menu = ['Prediction', 'What is CNN', 'Modeling Results','Team Mate']
choice = st.selectbox("Menu", menu)

# 메뉴에 따른 페이지 선택
if choice == 'Prediction':
    home()
elif choice == 'What is CNN':
    page1()
elif choice == 'Modeling Results':
    page2()
elif choice == 'Team Mate':
    Team_Mate()
