import numpy as np
from scipy import io
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.python.keras.models import Sequential
import matplotlib.pylab as plt


class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))


w = 526
h = 26
c = 1
train_image_count = 100000
input_shape = (w, h, c)
learning_rate = 0.001
regularization_rate = 0.0001
category_count = 7 + 1
n_epoch = 100
mini_batch_size = 64

model = Sequential()

# Layer 1
model.add(Conv2D(32,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Layer 2
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 3
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 1)))

# Layer 4
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 1)))

# flatten
model.add(Flatten(input_shape=input_shape))

# fc layers
model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(regularization_rate)))
model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(regularization_rate)))
model.add(Dense(category_count, activation='softmax', kernel_regularizer=regularizers.l2(regularization_rate)))

# read image
root_path = r'D:\Projects\emotion_in_speech\Audio_Speech_Actors_01-24/'
mat_path = root_path + 'logfbank.mat'
digits = io.loadmat(mat_path)
X, y, z, sr, lengths = digits.get('feature_matrix'), digits.get('emotion_label')[0], digits.get('intensity_label')[0], \
                       digits.get('sample_rate')[0], digits.get('actual_length')[0]  # X: nxm: n=1440//sample, m=feature
X = np.expand_dims(X,3)
y = y - 1
# X = X[:, ::100]

# select intensy
# for i in range(len(X)):

# n_samples, n_features = X.shape
train_data, test_data, train_label, test_label = train_test_split(X, y, test_size=0.2, random_state=777)

x_train = train_data
y_train = train_label
x_val = test_data
y_val = test_label
x_test = test_data
y_test = test_label

# reshape the data into a 4D tensor - (sample_number, x_img_size, y_img_size, num_channels)
# because the MNIST is greyscale, we only have a single channel - RGB colour images would have 3
# x_train = x_train.reshape(x_train.shape[0], w, h, c)
# x_val = x_val.reshape(x_val.shape[0], w, h, c)
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_val.shape[0], 'val samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices - this is for use in the
# categorical_crossentropy loss below
y_train = keras.utils.to_categorical(y_train, category_count)
y_val = keras.utils.to_categorical(y_val, category_count)
y_test = keras.utils.to_categorical(y_test, category_count)

# train
import tensorflow as tf
from tensorflow.python.keras.backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
# train
history = AccuracyHistory()
model.compile(loss=keras.losses.categorical_crossentropy,
              # optimizer=keras.optimizers.SGD(lr=0.01),
              optimizer=keras.optimizers.Adam(lr=learning_rate),
              metrics=['accuracy'])
model.summary()
model.fit(x_train, y_train,
          batch_size=mini_batch_size,
          epochs=n_epoch,
          verbose=2,
          validation_data=(x_val, y_val),
          callbacks=[history])
model.save_weights('../weights/' + '/model_weight.h5')
model.save('../weights/' + '/model.h5')
score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
plt.plot(range(1, n_epoch + 1), history.acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()
