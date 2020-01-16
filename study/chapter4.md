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

    10 0.9771633
    20 0.9748626
    30 0.97259146
    40 0.9703488
    50 0.968134
    60 0.9658969
    70 0.9637122
    80 0.9615781
    90 0.95939904
    100 0.957316
    예측값: [0 2 2 0 0 2]
    실제값: [0 1 2 0 0 2]
    


```python
is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도: %.2f' % sess.run(accuracy * 100, feed_dict = {X: x_data, Y:y_data}))
```

    정확도: 83.33
    
