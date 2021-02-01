import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np

# 读入数据集
x_data = datasets.load_iris().data
y_data = datasets.load_iris().target
# 数据集乱序
np.random.seed(116)  # 使用相同seed，使输入特征/标签一一对应
np.random.shuffle(x_data)
np.random.seed(116)
np.random.shuffle(y_data)
tf.random.set_seed(116)

# 训练集测试集划分
x_train = x_data[:-30]
y_train = y_data[:-30]
x_test = x_data[-30:]
y_test = y_data[-30:]

# 数据类型转换
x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)

# 配成【输入特征，标签】对，每次喂入一个batch
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# 定义神经网络中所有可训练参数
w1 = tf.Variable(tf.random.truncated_normal([4, 3], stddev=0.1, seed=1))
b1 = tf.Variable(tf.random.truncated_normal([3], stddev=0.1, seed=1))

lr = 0.1  # 学习率为0.1
train_loss_results = []  # 将每轮的loss记录在此列表中，为后续画loss曲线提供数据
test_acc = []  # 将每轮的acc记录在此列表中，为后续画acc提供数据
epoch = 500  # 循环500轮
loss_all = 0  # 每轮分 4 个step,loss_all记录四个step生成 4 分loss的和

# 嵌套循环迭代，with结构更新参数，显示当前loss
for epoch in range(epoch):  # 数据集级别的循环。每个epoch循环一次数据集
    for step, (x_train, y_train) in enumerate(train_db):  # batch级别的循环，每个step循环一个batch
        with tf.GradientTape() as tape:  # with结构记录梯度信息
            y = tf.matmul(x_train, w1) + b1
            y = tf.nn.softmax(y)
            y_ = tf.one_hot(y_train, depth=3)  # 将标签值转换为独热码格式
            loss = tf.reduce_mean(tf.square(y_ - y))
            loss_all += loss.numpy()
        # 计算loss对各参数的梯度
        grads = tape.gradient(loss, [w1, b1])

        # 实现梯度更新
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])

    # 每个epoch,打印loss信息
    print("Epoch {}, loss:{}".format(epoch, loss_all / 4))
    train_loss_results.append(loss_all / 4)  # 将4个step的loss求平均记录在此变量中
    loss_all = 0

    # 测试部分
    # total_correct 为预测对的样本个数，total_number 为预测的总测试样本数，将这两个变量初始化为0
    total_correct, total_number = 0, 0
    for x_test, y_test in test_db:
        # 使用更新后的参数进行预测
        y = tf.matmul(x_test, w1) + b1
        y = tf.nn.softmax(y)
        pred = tf.argmax(y, axis=1)
        pred = tf.cast(pred, dtype=y_test.dtype)
        # 若分类正确，correct=1，否则为0 。将bool转换为int型
        correct = tf.cast(tf.equal(pred, y_test), dtype=tf.int32)
        # 将每个batch的correct数加起来
        correct = tf.reduce_sum(correct)
        # 将所有batch中的correct数加起来
        total_correct += int(correct)
        # total_number为测试的样本数，也就是x_test的行数，shape[0]返回变量的行数
        total_number += x_test.shape[0]

    # 总的准确率等于total_correct/total_number
    acc = total_correct / total_number
    test_acc.append(acc)
    print("test_acc:", acc)
    print("-------------------------")

plt.title("Loss Function Curve")  # 标题
plt.xlabel("epoch")
plt.ylabel('Loss')
plt.plot(train_loss_results, label="$Loss$")  # 逐点画出trian_loss_results值并连线，连线图标是Loss
plt.legend()  # 画出图像图标
plt.show()

plt.title("Acc Curve")
plt.xlabel('Epoch')
plt.ylabel("Acc")
plt.plot(test_acc, label="$Accuracy$")
plt.legend()
plt.show()