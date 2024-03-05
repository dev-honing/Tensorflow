# main.py
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

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

# 배열의 모양 확인을 위한 콘솔 출력
# print(x_train.shape) # (60000, 28, 28, 1)
# print(x_test.shape) # (10000, 28, 28, 1)

# 데이터 세트 생성(.shuffle 메서드로 섞기)
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# 섞인 데이터 세트 객체 확인을 위한 콘솔 출력(여기서 None은 데이터 세트의 배치 크기가 유동적으로 조정됨을 의미)
# print(train_ds) 
# print(test_ds) 

# 모델 정의(tf.keras 모델)
class MyModel(Model):
    # 생성자 메서드로 모델의 레이어 초기화
    def __init__(self):
        super(MyModel, self).__init__()
        # 모델의 레이어 정의(Conv2D, Flatten, Dense)
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10)
    
    # 모델의 실행 메서드 정의(순전파 과정)
    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)
    
# 모델 인스턴스 생성
model = MyModel()

# 손실함수와 옵티마이저 선택
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

# 손실 및 성능 지표 설정
# 훈련 데이터에 대한 평균 손실 및 정확도 지표 설정
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

# 테스트 데이터에 대한 평균 손실 및 정확도 지표 설정
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

@tf.function
# 훈련 스텝 함수: 주어진 이미지 및 라벨로 모델을 훈련하고 손실 및 성능 지표 업데이트
def train_step(images, labels):
    # 계산을 위한 Gradient 컨텍스트 생성
    with tf.GradientTape() as tape:
        # 모델에 이미지 전달 및 예측
        predictions = model(images, training=True)
        # 손실 계산
        loss = loss_object(labels, predictions)
    
    # 손실에 대한 모델의 그래디언트 계산
    gradients = tape.gradient(loss, model.trainable_variables)

    # 옵티마이저를 사용해 가중치 업데이트
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # 훈련 데이터에 대한 손실 및 정확도 지표 업데이트
    train_loss(loss)
    train_accuracy(labels, predictions)

@tf.function
# 테스트 스텝 함수: 주어진 이미지 및 라벨로 모델을 평가하고 손실 및 성능 지표 업데이트
def test_step(images, labels):
    # 모델에 이미지 전달 및 예측
    predictions = model(images, training=False)
    # 손실 계산
    t_loss = loss_object(labels, predictions)

    # 테스트 데이터에 대한 손실 및 정확도 지표 업데이트
    test_loss(t_loss)
    test_accuracy(labels, predictions)