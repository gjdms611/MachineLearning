# CHAPTER 4 기본 신경망 구현
## 목차
1. [인공신경망의 작동 원리](#인공신경망의-작동-원리)<br>
2. [간단한 분류 모델 구현하기](#간단한-분류-모델-구현하기)<br>
3. [심층신경망 구현하기](#심층신경망-구현하기)<br><br>


### 인공신경망의 작동 원리

미래의 나야 작동원리 설명을 여기에 써줘

### 간단한 분류 모델 구현하기


```python
import tensorflow.compat.v1 as tf
import numpy as np
tf.disable_v2_behavior()

# [털, 날개] : 털/날개가 있는지없는지
x_data = np.array([[0,0],[1,0],[1,1],[0,0],[0,0],[0,1]])

y_data = np.array([
    [1,0,0], # 기타
    [0,1,0], # 포유류
    [0,0,1], # 조류
    [1,0,0],
    [1,0,0],
    [0,0,1]
])

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# 1.은 int가 아닌 float이라는 의미.
W = tf.Variable(tf.random_uniform([2,3], -1., 1.))
b = tf.Variable(tf.zeros([3]))

L = tf.add(tf.matmul(X,W), b)
L = tf.nn.relu(L)
```

    WARNING:tensorflow:From c:\users\infinite\appdata\local\programs\python\python37\lib\site-packages\tensorflow_core\python\compat\v2_compat.py:65: disable_resource_variables (from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.
    Instructions for updating:
    non-resource variables are not supported in the long term
    

여기서 논란이 되었던 것은 x_data * W는 [6,3]인데 어떻게 [3]과의 덧셈 연산을 지원하는가였다.<br>
결론은 feed_dict를 사용해 매개변수를 넘겨주게 되면 원소를 자동으로 분리하여, [6,2]로 넘겨주면 6개의 [2]를 전달한다고 한다!


```python
model = tf.nn.softmax(L)
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(model), axis=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train_op = optimizer.minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in range(100):
    sess.run(train_op, feed_dict={X:x_data, Y:y_data})
    
    if (step + 1)%10 ==0:
        print(step+1, sess.run(cost, feed_dict = {X:x_data, Y:y_data}))
        
prediction = tf.argmax(model, axis=1)
target = tf.argmax(Y,axis=1)
print('예측값:', sess.run(prediction, feed_dict={X:x_data}))
print('실제값:', sess.run(target, feed_dict={Y:y_data}))
```

    10 1.0942008
    20 1.0848997
    30 1.0758835
    40 1.0671607
    50 1.0586861
    60 1.0504823
    70 1.0425204
    80 1.0348142
    90 1.0273567
    100 1.0200877
    예측값: [0 2 2 0 0 2]
    실제값: [0 1 2 0 0 2]
    

argmax는 주어진 행렬에서 가장 큰 값의 인덱스를 반환하는 함수이다.<br>
tf.argmax(a, axis)에서 aixs는 기준을 정해준다.<br>
axis는 차원에 따라 지정해야 할 값이 달라진다. 1차원은 지정할 필요가 없고, 2차원일 경우 0 or 1, 3차원일 경우 0, 1, 2...<br>
따라서 axis = 1일 경우 행을 기준으로 가장 큰 값의 인덱스를 리턴한다.

```python
is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도: %.2f' % sess.run(accuracy * 100, feed_dict = {X: x_data, Y:y_data}))
```

    정확도: 83.33
    

### 심층 신경망 구현하기


```python
W1 = tf.Variable(tf.random_uniform([2,10], -1., 1.))
W2 = tf.Variable(tf.random_uniform([10,3], -1., 1.))

b1 = tf.Variable(tf.zeros([10]))
b2 = tf.Variable(tf.zeros([3]))

L1 = tf.add(tf.matmul(X, W1), b1)
L1 = tf.nn.relu(L1)

model = tf.add(tf.matmul(L1, W2), b2)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = Y, logits=model))

optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost)

init = tf.global_variables_initializer()
sess.run(init)

for step in range(100):
    sess.run(train_op, feed_dict={X: x_data, Y: y_data})
    
    if (step + 1)%10 ==0:
        print(step+1, sess.run(cost, feed_dict = {X:x_data, Y:y_data}))
        
prediction = tf.argmax(model, 1)
target = tf.argmax(Y,1)
print('예측값:', sess.run(prediction, feed_dict={X: x_data}))
print('실제값:', sess.run(target, feed_dict = {Y: y_data}))

is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도 : %.2f' % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}))
```

    10 0.76788783
    20 0.62280303
    30 0.50967103
    40 0.4199364
    50 0.34823382
    60 0.29070586
    70 0.24296229
    80 0.20615202
    90 0.17746647
    100 0.15469693
    예측값: [0 1 2 0 0 2]
    실제값: [0 1 2 0 0 2]
    정확도 : 100.00
    
