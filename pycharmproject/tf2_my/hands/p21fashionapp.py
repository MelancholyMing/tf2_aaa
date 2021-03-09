from PIL import Image
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

type = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

model_save_path = "./fashion_checkpoint/fashion.ckpt"
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.load_weights(model_save_path)
preNum = int(input("number of test picture:"))
for i in range(preNum):
    image_path = input("the path of image:")
    image = plt.imread(image_path)
    plt.set_cmap('gray')
    plt.imshow(image)

    img = Image.open(image_path)
    img = img.resize((28, 28), Image.ANTIALIAS)
    img_arr = np.array(img.convert("L"))
    img_arr = 255 - img_arr
    img_arr = img_arr / 255.0

    x_predict = img_arr[tf.newaxis, ...]
    result = model.predict(x_predict)
    pred = tf.argmax(result, axis=1)
    print('\n')
    tf.print(type[int(pred)])

    plt.pause(1)
    plt.close()