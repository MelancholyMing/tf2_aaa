import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

input_word = 'abcde'
w_to_id = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}

x_train = [w_to_id[i] for i in input_word]
y_train = [w_to_id['b'], w_to_id['c'], w_to_id['d'], w_to_id['e'], w_to_id['a']]

np.random.seed(7)
np.random.shuffle(x_train)
np.random.seed(7)
np.random.shuffle(y_train)
tf.random.set_seed(7)

x_train = np.reshape(x_train, (len(x_train), 1))  # 使x_train符合Embedding输入要求：[送入样本数， 循环核时间展开步数]
y_train = np.array(y_train)

model = tf.keras.models.Sequential([
    Embedding(5, 2),  # [词汇表大小, 编码维度]
    SimpleRNN(3),
    Dense(5, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

model_savepath = "./rnn_embedding_checkpoint/rnn_embedding_1pre1.ckpt"
if os.path.exists(model_savepath + '.index'):
    print("=================load model==================")
    model.load_weights(model_savepath)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_savepath,
                                                 save_weights_only=True,
                                                 save_best_only=True,
                                                 monitor='loss')
history = model.fit(x_train, y_train, batch_size=32, epochs=5,
                    callbacks=[cp_callback])

model.summary()

with open('rnn_embedding_1pre1_weights.txt', 'w') as f:
    for v in model.trainable_variables:
        f.write(str(v.name) + '\n')
        f.write(str(v.shape) + '\n')
        f.write(str(v.numpy()) + '\n')

acc = history.history['sparse_categorical_accuracy']
loss = history.history['loss']

plt.plot(acc, label='training accuracy')
plt.plot(loss, label='training loss')
plt.title("training accuracy and loss")
plt.legend()
plt.show()

preNum = int(input('input the number of test alphabet:'))
for i in range(preNum):
    alphabet1 = input('input testing alphabet:')
    alphabet = w_to_id[alphabet1]
    alphabet = np.reshape(alphabet, (1, 1))
    result = model.predict(alphabet)
    pred = int(tf.argmax(result, axis=1))
    tf.print(alphabet1 + "->" + input_word[pred])

