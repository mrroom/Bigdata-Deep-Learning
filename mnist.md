```python
import numpy as np

X = np.array([1,2])
print(X)

W = np.array([[1,3,5],[2,4,6]])
print(W)
print(W.ndim) # 차원:Row
print(W.shape) # (행수,열수) 
print(W.size) # 전체 원소 데이터 수
Z = np.dot(X,W) # 내적 => X:행수, W:열수
print(Z)
```

    [1 2]
    [[1 3 5]
     [2 4 6]]
    2
    (2, 3)
    6
    [ 5 11 17]
    


```python
from matplotlib import pyplot as plt

# 시그모이드 함수 : 분류(>1)
def sigmoid(x):
    return 1/(1+np.exp(-x))


# 첫번째 은닉층 계산
X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]) # 첫번째 은닉층  가중치
B1 = np.array([0.1, 0.2, 0.3]) # bias 편향

print(W1.shape)
print(X.shape)
print(B1.shape)

A1 = np.dot(X, W1) + B1 #노드 * 가중치 + 편향

print(A1)

Z1 = sigmoid(A1) #첫번째 활성화 함수
print(Z1)

# 두번째 은닉층 계산

W2 = np.array([[0.1, 0.4],[0.2,0.5],[0.3,0.6]]) # 두번째 은닉층  가중치
B2 = np.array([0.1, 0.2]) # 두번째 bias 편향

print(Z1.shape)
print(W2.shape)
print(B2.shape)

A2 = np.dot(Z1, W2) + B2
print(A2)

Z2 = sigmoid(A2) #두번째 활성화 함수
print(Z2)

def identity_function(x):
    return x

# 출력층 계산
 
W3 = np.array([[0.1, 0.3],[0.2,0.4]]) # 출력층 가중치
B3 = np.array([0.1,0.2]) # 출력층 bias 편향

A3 = np.dot(Z2, W3)+B3

Z3 = identity_function(A3)

print(Z3)

def init_network():
    network = {}
    network['W1'] = np.array([[0.1,0.3,0.5],[0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1,0.2,0.3])
    network['W2'] = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
    network['b2'] = np.array([[0.1,0.2]])
    network['W3'] = np.array([[0.1,0.3],[0.2,0.4]])
    network['b3'] = np.array([[0.1,0.2]])
    
    return network

def foward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)
    
    return y

network = init_network()
x = np.array([1.0, 0.5])
y = foward(network, x)
print(y)


```

    (2, 3)
    (2,)
    (3,)
    [0.3 0.7 1.1]
    [0.57444252 0.66818777 0.75026011]
    (3,)
    (3, 2)
    (2,)
    [0.51615984 1.21402696]
    [0.62624937 0.7710107 ]
    [0.31682708 0.69627909]
    [[0.31682708 0.69627909]]
    


```python
import sys, os
sys.path.append("./dataset") 
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)
(60000, 784)
(60000,)
(10000, 784)
(10000,)


def img_show(img):
    return Image.fromarray(np.uint8(img))
    
img = x_train[0]
label = t_train[0]
print(label)

print(img.shape)
img = img.reshape(28, 28)
print(img.shape)

img_show(img)


import pickle

# 시그모이드 함수
def sigmoid(x):
    return 1/(1+np.exp(-x))

# 소프트맥스 함수
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    
    return 
        
# 테스트 데이터(테스트 데이터, 확률값) 가져오기
def get_data():
    # normalize:정규화(0~1), flatten:1차원데이터로 펼침, one_hot_label:정담만1이고 나머지0으로 만든다
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

# 층별 노드의 가중치와 편향 가져오기
def init_network():
    with open("dataset/sample_weight.pkl", "rb") as f:
    #with open("dataset/mnist.pkl", "rb") as f:
        network = pickle.load(f)
        
    return network

# 예측 작업
def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1) #분류(0,1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2) #분류(0,1)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)  #분류(1>)
    
    return y

# 테스트 데이터(테스트 데이터, 확률값) 가져오기
x, t = get_data()
# 층별 노드의 가중치와 편향 가져오기
network = init_network()

accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])#테스트 데이터로 예측
    p = np.argmax(y) #예측결과의 확률값
    if p == t[i]:
        accuracy_cnt += 1
    
print("Accuracy:" + str(float(accuracy_cnt)/len(x)))
print('predicted:',p,',','real:',t[i])

```

    (60000, 784)
    (60000,)
    (10000, 784)
    (10000,)
    5
    (784,)
    (28, 28)
    Accuracy:0.098
    predicted: 0 , real: 6
    


```python
# 평균제곱오차 Mean Squared Error

def mean_squared_error(y, t):
    return 0.5*np.sum((y-t)**2)

#정답은 2
t = [0,0,1,0,0,0,0,0,0,0]

y1 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
print(mean_squared_error(np.array(y1), np.array(t)))

y2 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
print(mean_squared_error(np.array(y2), np.array(t)))

```

    0.09750000000000003
    0.5975
    


```python
import sys, os 
sys.path.append(os.pardir) 
import numpy as np 
import pickle 
from dataset.mnist import load_mnist 
from common.functions import sigmoid, softmax 

# 테스트 데이터(테스트 데이터, 확률값) 가져오기
def get_data(): 
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False) 
    return x_test, t_test 

# 층별 노드의 가중치와 편향 가져오기
def init_network(): 
    with open("dataset/sample_weight.pkl", 'rb') as f: 
        network = pickle.load(f) 
    return network 

# 예측 작업
def predict(network, x): 
    W1, W2, W3 = network['W1'], network['W2'], network['W3'] 
    b1, b2, b3 = network['b1'], network['b2'], network['b3'] 

    a1 = np.dot(x, W1) + b1 
    z1 = sigmoid(a1) #분류(0,1)
    a2 = np.dot(z1, W2) + b2 
    z2 = sigmoid(a2) #분류(0,1)
    a3 = np.dot(z2, W3) + b3 
    y = softmax(a3) #분류(>1)

    return y 

# 테스트 데이터(테스트 데이터, 확률값) 가져오기
x, t = get_data() 
# 층별 노드의 가중치와 편향 가져오기
network = init_network() 

accuracy_cnt = 0 
for i in range(len(x)): 
    y = predict(network, x[i])  #테스트 데이터로 예측
    p= np.argmax(y)  #예측결과의 확률값
    if p == t[i]: 
        accuracy_cnt += 1 

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
print('predicted:',p,',','real:',t[i])
```

    Accuracy:0.9352
    predicted: 6 , real: 6
    


```python
# 교차 엔트로피 오차 Cross Entropy Error

def cross_entropy_error(y,t):
    delta = 1e-7
    return -np.sum(t * np.log(y+delta))
# y + delta인 이유는 만약 y가 0일 때는 infinite 값을 반환하므로 계산이 안되기 때문에 아주 작은 임의의 값을 입력하여 값이 -inf가 되는 것을 막는다



t = [0,0,1,0,0,0,0,0,0,0]

y1 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
print(cross_entropy_error(np.array(y1), np.array(t)))

y2 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
print(cross_entropy_error(np.array(y2), np.array(t)))
```

    0.510825457099338
    2.302584092994546
    
