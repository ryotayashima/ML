from keras.datasets import imdb
from keras import models
from keras import layers
from keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt

def vectorize_sequence(sequence, dimension= 10000):
    results = np.zeros((len(sequence), dimension))      # (len(sequence), dimension) の零行列
    for i, sequence in enumerate(sequence):     # results[i]のインデックスを1に設定
        results[i, sequence] = 1.
    return results

'''
データの設定
'''
# data:レビューのリスト(単語のシーケンスをエンコード済み)
# labels: 0で「否定的」、1で「肯定的」
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)      # 出現率の高い上位10000語を選択

x_train = vectorize_sequence(train_data)        # 訓練データのベクトル化
x_test = vectorize_sequence(test_data)          # テストデータのベクトル化
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

'''
モデルの構築
'''
early_stopping = EarlyStopping(monitor= "val_loss", patience= 10, verbose= 1)
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation="relu"))
model.add(layers.Dense(16, activation="relu"))
model.add(layers.Dense(16, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))
model.compile(optimizer="rmsprop",loss="binary_crossentropy",metrics=["accuracy"])

'''
モデル学習
'''
history = model.fit(partial_x_train,partial_y_train,epochs=20,batch_size=512,validation_data=(x_val, y_val), callbacks= [early_stopping])

'''
予測精度の評価
'''
history_dict = history.history
loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]
acc_values = history_dict["acc"]
val_acc_values = history_dict["val_acc"]

epochs = range(1, len(loss_values) + 1)
plt.rc('font', family='serif')
fig = plt.figure()
ax_loss = fig.add_subplot(111)
g1 = ax_loss.plot(epochs, loss_values, label='loss', color='gray')
g2 = ax_loss.plot(epochs, val_loss_values, label= "val_loss", color="blue")
ax_acc = ax_loss.twinx()
g3 = ax_acc.plot(epochs, acc_values, label='acc', color='pink')
g4 = ax_acc.plot(epochs, val_acc_values, label='val_acc', color='red')
plt.title("loss & acc")

h1, l1 = ax_loss.get_legend_handles_labels()
h2, l2 = ax_acc.get_legend_handles_labels()
ax_loss.legend(h1+h2, l1+l2, loc='upper center',bbox_to_anchor=(0.5,-0.15),ncol=4)

ax_loss.set_xlabel("Epochs")
ax_loss.set_ylabel("loss")
ax_loss.grid(True)
ax_acc.set_ylabel("acc")
plt.show()










