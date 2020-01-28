# CHAPTER 5 텐서보드와 모델 재사용
## 목차
1. [학습 모델 저장하고 재사용하기](#학습-모델-저장하고-재사용하기)<br>
2. [텐서보드 사용하기](#텐서보드-사용하기)<br>

### 학습 모델 저장하고 재사용하기
우선 효율적인 코딩을 위해 데이터 파일을 따로 저장하는것으로 한다.</br>
data.csv에 우리가 사용할 데이터를 넣어 저장해둔다.</br>
그리고 .py파일에 사용할 코드를 작성한다.
```
import tensorflow.compat.v1 as tf
import numpy as np
tf.disable_v2_behavior()

data = np.loadtxt('./data.csv', delimiter=',', unpack=True, dtype='float32')
x_data = np.transpose(data[0:2])
y_data = np.transpose(data[2:])
```
numpy라이브러리를 이용해 텍스트 파일을 불러와 행렬로 변환시켜서 각각 저장하였다.
```
global_step = tf.Variable(0, trainable=False, name='global_step')

X=tf.placeholder(tf.float32)
Y=tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_uniform([2,10], -1., 1.))
L1 = tf.nn.relu(tf.matmul(X, W1))

W2 = tf.Variable(tf.random_uniform([10,20], -1., 1.))
L2 = tf.nn.relu(tf.matmul(L1, W2))

W3 = tf.Variable(tf.random_uniform([20, 3], -1, 1))
model = tf.matmul(L2, W3)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=model))

optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost, global_step=global_step)
```
학습을 진행할 때마다 global_step을 1씩 더해서 학습 횟수를 카운트할 수 있다.</br>
학습에 직접적으로 사용되지는 않는다.
```
sess = tf.Session()
saver = tf.train.Saver(tf.global_variables())
```
tf.global_variables()를 통해 앞서 정의했던 변수를 불러온다. 이를 통해 마지막으로 최적화가 진행되었던 변수를 가져와 마지막으로 학습이 진행되었던 단계에서부터 학습을 시작할 수 있다.
```
ckpt = tf.train.get_checkpoint_state('./model')
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    sess.run(tf.global_variables_initializer())
```
이전에 학습했던 모델이 존재할 경우 saver.restore()를 이용해 학습된 값을 불러오고, 그렇지 않을 경우 변수들을 초기화해주는 코드라고 한다.</br>
model폴더 내부에서 기존에 학습했던 모델을 가져오기 때문에 model폴더가 미리 있어야 한다.</br>
이걸 보지 않고 코딩하라고 하면.. 외울 수 있을지..
```
for step in range(2):
    sess.run(train_op, feed_dict={X: x_data, Y:y_data})

    print('Step: %d, ' % sess.run(global_step), 'Cost: %.3f' % sess.run(cost, feed_dict={X: x_data, Y:y_data}))

saver.save(sess, './model/dnn.ckpt', global_step=global_step)
```
학습이 끝난 후 학습된 변수들을 체크포인트 파일에 저장한다.
```
prediction = tf.argmax(model, 1)
target = tf.argmax(Y, 1)
print('예측값:', sess.run(prediction, feed_dict={X:x_data}))
print('실제값:', sess.run(target, feed_dict={Y:y_data}))

is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도: %.2f' % sess.run(accuracy * 100, feed_dict={X:x_data, Y:y_data}))
```
실행 결과를 살펴보면 학습이 진행될 수록 Cost값이 줄어들며 정확도가 올라가는 것을 확인할 수 있다.
```
Step: 1,  Cost: 1.525
Step: 2,  Cost: 1.371
예측값: [0 0 2 0 0 2]
실제값: [0 1 2 0 0 2]
정확도: 83.33

Step: 3,  Cost: 1.239
Step: 4,  Cost: 1.131
예측값: [0 0 2 0 0 2]
실제값: [0 1 2 0 0 2]
정확도: 83.33

Step: 5,  Cost: 1.045
Step: 6,  Cost: 0.980
예측값: [0 0 2 0 0 2]
실제값: [0 1 2 0 0 2]
정확도: 83.33

Step: 7,  Cost: 0.929
Step: 8,  Cost: 0.891
예측값: [0 1 2 0 0 2]
실제값: [0 1 2 0 0 2]
정확도: 100.00
```
다만 한가지 아쉬운(?)점은 가장 처음 생성되는 변수값은 랜덤하기 때문에 처음 실험을 시작할 때의 변수값에 따라 정확도가 100%가 되는 시기가 차이가 난다는 점이다. 만일 진짜로 여러 모델의 학습 속도를 측정하기 위해서는 가장 처음으로 들어가는 변수의 값을 정해놓고 실험해야 하지 않을까 하고 생각해본다. 물론 실제로 사용할 때는 우리가 준비한 데이터값을 넣으면서 하니까 학습 과정에서의 데이터가 랜덤한 것은 크게 중요하지 않을거라고 생각하고,,,</br></br>

### 텐서보드 사용하기
텐서보드를 사용하기 위해서는 우리가 사용한 그래프들을 with tf.name_scope를 사용하여 계층별로 묶어주어야 한다.</br>
```
with tf.name_scope('layer1'):
    W1 = tf.Variable(tf.random_uniform([2,10], -1., 1.), name='W1')
    L1 = tf.nn.relu(tf.matmul(X, W1))

with tf.name_scope('layer2'):
    W2 = tf.Variable(tf.random_uniform([10,20], -1., 1.), name='W2')
    L2 = tf.nn.relu(tf.matmul(L1, W2))

with tf.name_scope('output'):
    W3 = tf.Variable(tf.random_uniform([20, 3], -1, 1), name='W3')
    model = tf.matmul(L2, W3)

with tf.name_scope('optimizer'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=model))

    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(cost, global_step=global_step)

    tf.summary.scalar('cost', cost)
```
![그래프](../image/graph.jpg)
![텐서그래프](../image/tensorgraph.jpg)