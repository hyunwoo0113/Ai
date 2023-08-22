import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# 이미지 불러오고 전처리하는 함수
def load_and_preprocess_image(image_path, target_size):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # 이미지를 [0, 1] 범위로 스케일링
    img_array = tf.expand_dims(img_array, axis=0)  # 배치 차원 추가
    return img_array

# 저장된 모델 불러오기
loaded_model = tf.keras.models.load_model('siamese_positive_model')

# 이미지 경로
image_path_a = '학생증2.jpg'
image_path_b = '학생증4.jpg'

# 이미지 크기
input_shape = (128, 128, 3)

# 이미지 불러오기 및 전처리
image_a = load_and_preprocess_image(image_path_a, input_shape)
image_b = load_and_preprocess_image(image_path_b, input_shape)

# 이미지 유사도 예측
similarity_score = loaded_model.predict([image_a, image_b])[0][0]
print('Similarity Score:', similarity_score)
