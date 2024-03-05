# Tensorflow
https://www.tensorflow.org/tutorials/quickstart/advanced?hl=ko
## 가상환경 세팅 및 실행

## Tensorflow 세팅
### 1. 버전 확인 및 콘솔 출력하기
```
import tensorflow as tf
print("Hello, Tensorflow! \n Tensorflow version: ", tf.__version__)
```
```
(.venv) C:\Users\Administrator\Desktop\ho\Tensorflow>python main.py
2024-03-05 11:17:20.635748: I tensorflow/core/util/port.cc:113] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
WARNING:tensorflow:From C:\Users\Administrator\Desktop\ho\Tensorflow\.venv\Lib\site-packages\keras\src\losses.py:2976: The name tf.losses.sparse_softmax_cross_entropy is deprecated. Please use tf.compat.v1.losses.sparse_softmax_cross_entropy instead.

Hello, Tensorflow!
 Tensorflow version:  2.15.0
```

## 용어 정리
### 1. 데이터 전처리
1. 정규화: 이미지 데이터는 0 ~ 255의 범위의 픽셀 값을 가진다. 정규화를 통해 각 픽셀 값을 0 ~ 1 사이로 변환해 모든 특성이 동일한 범위를 가지도록 한다. 이를 통해 모델의 학습 과정을 안정적으로 진행하고, 학습 속도를 향상한다.

2. 데이터 형식 변환: 이미지 데이터를 네트워크에 입력하기 위해 적절한 형식으로 변환이 필요하다. 일반적으로 이미지는 2차원 배열 형태로 표현되는데, 딥러닝 모델은 3차원 형태의 입력을 기대한다. 따라서 이미지 데이터를 {이미지 수, 이미지 높이, 이미지 너비, 채널 수}의 4차원 배열 형태로 변환한다. (MNIST 데이터는 흑백 이미지이므로 채널 수가 1이다.)

이러한 전처리 단계를 거쳐 데이터의 준비를 완료하면, 더 효율적인 학습 수행이 가능하며 좋은 성능을 기대할 수 있다.