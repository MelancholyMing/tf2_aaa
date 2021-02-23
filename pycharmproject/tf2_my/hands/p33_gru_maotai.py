import os
import math
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.layers import Dense, Dropout, GRU

maotai = pd.read_csv('SH600519.csv')
train_set = maotai.iloc[:2426 - 300, 2:3].values
test_set = maotai.iloc[2426 - 300:, 2:3].values

sc = MinMaxScaler(feature_range=(0, 1))
train_set_scale = sc.fit_transform(train_set)
test_set_scale = sc.transform(test_set)

x_train, y_train = [], []
x_test, y_test = [], []

for i in range(60, len(train_set_scale)):
    x_train.append(train_set_scale[i - 60:i, 0])
    y_train.append(train_set_scale[i, 0])

np.random.seed(7)
np.random.shuffle(x_train)
np.random.seed(7)
np.random.shuffle(y_train)
tf.random.set_seed(7)

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], 60, 1))

for i in range(60, len(test_set_scale)):
    x_test.append(test_set_scale[i - 60:i, 0])
    y_test.append(test_set_scale[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0], 60, 1))

model = tf.keras.models.Sequential([
    GRU(units=80, return_sequences=True),
    Dropout(0.2),
    GRU(100),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='mean_squared_error')

model_savepath = "./gru_maotai_checkpoint/gru_maotai.ckpt"
if os.path.exists(model_savepath + '.index'):
    print("===============load model===================")
    model.load_weights(model_savepath)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_savepath,
                                                 save_weights_only=True,
                                                 save_best_only=True,
                                                 monitor='val_loss')

history = model.fit(x_train, y_train, batch_size=64, epochs=50, validation_data=(x_test, y_test),
                    validation_freq=1, callbacks=[cp_callback])

model.summary()

with open('GRU_maotai_weights.txt', 'w') as f:
    for v in model.trainable_variables:
        f.write(str(v.name) + '\n')
        f.write(str(v.shape) + '\n')
        f.write(str(v.numpy()) + '\n')

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(loss, label='training loss')
plt.plot(val_loss, label='validation loss')
plt.title('training and validation loss')
plt.legend()
plt.show()

predicted_stock_price = sc.inverse_transform(model.predict(x_test))
real_price = test_set[60:]

plt.plot(real_price, color='red', label='maotai stock price')
plt.plot(predicted_stock_price, color='blue', label='predicted price')
plt.title("maotai stock prediction")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.legend()
plt.show()

mse = mean_squared_error(predicted_stock_price, real_price)
rmse = math.sqrt(mse)
mae = mean_absolute_error(predicted_stock_price, real_price)

print("均方误差：", mse)
print("均方根误差", rmse)
print("平均绝对值误差", mae)
