# main.py
import tensorflow as tf

# 버전 확인 및 콘솔 출력 테스트
# print("Hello, Tensorflow! \n Tensorflow version: ", tf.__version__)

# MNIST 데이터 로드
mnist = tf.keras.datasets.mnist

# 훈련 데이터 세트
"""
x_train: 훈련 데이터를 저장하는 변수. {이미지 수, 이미지 높이, 이미지 너비}의 3차원 배열
y_train: 각 이미지의 실제 숫자 라벨을 저장하는 변수

x_test: 테스트 이미지 데이터를 저장하는 변수. {이미지 수, 이미지 높이, 이미지 너비}의 3차원 배열
y_test: 각 이미지의 실제 숫자 라벨을 저장하는 변수
"""
# mnist.load_data()에서 반환된 데이터를 튜플에 저장
# 각 튜플은 훈련 데이터와 테스트 데이터를 나타냄
(x_train, y_train), (x_test, y_test) = mnist.load_data()
"""
훈련 데이터: 주어진 데이터에 대한 패턴을 학습한 데이터
테스트 데이터: 실제 정답이 라벨링된 학습에 사용되지 않은 데이터
"""

# 콘솔 출력 테스트
# print(mnist.load_data())

# MNIST 데이터 전처리
x_train, x_test = x_train / 255.0, x_test / 255.0

# 로드된 데이터 확인(.shape 메서드로 배열의 모양을 확인)
# print("훈련 데이터: ", x_train.shape)
# print("훈련 라벨: ", y_train.shape)
# print("테스트 데이터: ", x_test.shape)
# print("테스트 라벨: ", y_test.shape)
"""
훈련 데이터:  (60000, 28, 28) # 60000개의 이미지, 28x28 크기
훈련 라벨:  (60000,) # 60000개의 라벨
테스트 데이터:  (10000, 28, 28) # 10000개의 이미지, 28x28 크기
테스트 라벨:  (10000,) # 10000개의 라벨
"""

# 색상 채널 추가
x_train = x_train[..., tf.newaxis].astype("float32")
x_test = x_test[..., tf.newaxis].astype("float32")

print(x_train.shape) # (60000, 28, 28, 1)
print(x_test.shape) # (10000, 28, 28, 1)