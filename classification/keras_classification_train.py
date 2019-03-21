import numpy as np
from scipy import io
# import keras
from tensorflow import keras
from tensorflow.python.keras import regularizers, optimizers
from tensorflow.python.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Activation
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
# from tensorflow.python.keras.layers.normalization import BatchNormalization
import matplotlib.pylab as plt
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
import dataset_split
import model_parameter

h, w, kernel_size, kernel_stride, pool_stride, pool_size_list = model_parameter.get_parameter_149_26()
c = 2
train_image_count = 100000
input_shape = (h, w, c)
learning_rate = 0.0001
regularization_rate = 0.0001
category_count = 7 + 1
n_epoch = 300
mini_batch_size = 256

model = Sequential()

# Layer 1
model.add(Conv2D(32,
                 kernel_size=kernel_size,
                 strides=kernel_stride,
                 activation='relu',
                 input_shape=input_shape))
# model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(MaxPooling2D(pool_size=pool_size_list[0], strides=pool_stride))

# Layer 2
model.add(Conv2D(64, kernel_size, activation='relu'))
model.add(Dropout(0.3))
model.add(MaxPooling2D(pool_size=pool_size_list[1]))
# Layer 3
model.add(Conv2D(128, kernel_size, activation='relu'))
model.add(Dropout(0.3))
model.add(MaxPooling2D(pool_size=pool_size_list[2]))

# Layer 4
model.add(Conv2D(256, kernel_size, activation='relu'))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(MaxPooling2D(pool_size=pool_size_list[3]))

# flatten
model.add(Flatten(input_shape=input_shape))

# fc layers
model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(regularization_rate)))
model.add(Dropout(0.4))
model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(regularization_rate)))
model.add(Dropout(0.4))
model.add(Dense(category_count, activation='softmax', kernel_regularizer=regularizers.l2(regularization_rate)))

# read image
root_path = r'D:\Projects\emotion_in_speech\Audio_Speech_Actors_01-24/'
mat_file_name = 'mfcc_logf_slice_150_025.mat'
mat_path = root_path + mat_file_name
digits = io.loadmat(mat_path)
split_method = 'rep'

# X: nxm: n=1440//sample, m=feature
# X = np.expand_dims(X,3)
# X = X[:, ::100]

# select intensy
# for i in range(len(X)):

# n_samples, n_features = X.shape
# train_data, test_data, train_label, test_label = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=777)
# train_data, test_data, train_label, test_label = train_test_rep_split(X, y, rep)
train_data, train_label, test_data, test_label, normal_test_sets, strong_test_sets = \
    dataset_split.train_test_rep_split4(digits, c, split_method)

x_train = train_data
y_train = train_label
x_val = test_data[:2000]
y_val = test_label[:2000]
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
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
# train
model.compile(loss=keras.losses.categorical_crossentropy,
              # optimizer=keras.optimizers.SGD(lr=0.01),
              optimizer=optimizers.RMSprop(lr=learning_rate, decay=1e-5),
              metrics=['accuracy'])
model_save_path = root_path + '/model_' + mat_file_name.split('.')[0] + '_' + split_method + '.h5'
checkpoint = ModelCheckpoint(model_save_path, monitor='val_acc', verbose=1,
                             save_best_only=True, save_weights_only=False, mode='auto', period=1)

earlystop = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=0, mode='auto',
                          restore_best_weights=True)
callback_list = [checkpoint, earlystop]
model.summary()
history = model.fit(x_train, y_train,
                    batch_size=mini_batch_size,
                    epochs=n_epoch,
                    verbose=2,
                    validation_data=(x_val, y_val),
                    callbacks=callback_list)
# model.save_weights(root_path + '/weight_' + mat_file_name.split('.')[0] + '_' + split_method + '.h5')
# model.save(model_save_path)
score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Plot training & validation accuracy values
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
acc = ax1.plot(history.history['acc'], label='Train acc')
val_acc = ax1.plot(history.history['val_acc'], label='Val acc')
loss = ax2.plot(history.history['loss'], label='Train loss')
val_loss = ax2.plot(history.history['val_loss'], label='Val loss')
lines = acc + val_acc + loss + val_loss
labs = [l.get_label() for l in lines]
ax1.legend(lines, labs, loc=0)
plt.title('Model accuracy')
ax1.set_ylabel('Accuracy')
ax2.set_ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

# intensity test
emotion_list = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
for i in range(len(normal_test_sets)):
    test_set = np.array(normal_test_sets[i])
    y_test = keras.utils.to_categorical(np.array([i] * len(test_set)), category_count)
    score = model.evaluate(test_set, y_test, verbose=0)
    print('{} Test accuracy:\t{}'.format(emotion_list[i], score[1]))

for i in range(len(strong_test_sets) - 1):
    i += 1
    test_set = np.array(strong_test_sets[i])
    y_test = keras.utils.to_categorical(np.array([i] * len(test_set)), category_count)
    score = model.evaluate(test_set, y_test, verbose=0)
    print('{} Test accuracy:\t{}'.format(emotion_list[i], score[1]))