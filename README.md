# Tensorflow

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

