import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

# 이미지 불러오고 전처리하는 함수
def load_and_preprocess_image(image_path, target_size):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # 이미지를 [0, 1] 범위로 스케일링
    img_array = tf.expand_dims(img_array, axis=0)  # 배치 차원 추가
    return img_array

# 올바른 이미지 쌍이 있는 디렉토리 경로 설정
positive_image_directory = 'reference_images'  # 올바른 이미지 쌍이 있는 디렉토리 경로로 수정

# 이미지 파일 경로들 생성
image_paths = [os.path.join(positive_image_directory, filename) for filename in os.listdir(positive_image_directory)]

# 이미지 크기
input_shape = (128, 128, 3)

# 데이터 로딩 및 전처리
image_data = []
for image_path in image_paths:
    image = load_and_preprocess_image(image_path, input_shape)
    image_data.append(image)

# 모델 구성
def build_siamese_model(input_shape):
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    base_network = tf.keras.Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=input_shape),
        Flatten(),
        Dense(128, activation='relu')
    ])

    encoded_a = base_network(input_a)
    encoded_b = base_network(input_b)

    similarity_score = tf.keras.layers.Dot(axes=1, normalize=True)([encoded_a, encoded_b])

    siamese_model = Model(inputs=[input_a, input_b], outputs=similarity_score)
    return siamese_model

# Siamese Network 모델 생성
siamese_model = build_siamese_model(input_shape)
siamese_model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# 학습 데이터 생성
train_data = [np.concatenate(image_data), np.concatenate(image_data)]
train_labels = np.ones(len(image_data))  # 올바른 이미지 쌍의 경우 레이블 1

# 모델 학습
siamese_model.fit(train_data, train_labels, epochs=10, batch_size=32, validation_split=0.2)

# 학습된 모델 저장
siamese_model.save('siamese_positive_model')
siamese_model.summary()
