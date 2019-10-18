import numpy as np
from scipy import io
from tensorflow import keras
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# from tensorflow.python.keras.layers.normalization import BatchNormalization
import matplotlib.pylab as plt
from sklearn import model_selection
import dataset_split
import model_parameter
import logging
import sys


def get_cnn_model():
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
    model.add(Flatten())
    # fc layers
    model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(regularization_rate)))
    model.add(Dropout(0.4))
    model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(regularization_rate)))
    model.add(Dropout(0.4))
    model.add(Dense(category_count, activation='softmax', kernel_regularizer=regularizers.l2(regularization_rate)))

    return model


# Disable Tensorflow debugging information
logging.getLogger("tensorflow").setLevel(logging.ERROR)

train_image_count = 100000
learning_rate = 0.0001
regularization_rate = 0.00001
category_count = 7 + 1
n_epoch = 500
mini_batch_size = 256

# read image
root_path = r'D:\Projects\emotion_in_speech\vis_mat/'
mat_file_name = 'mfcc_logf.mat'
split_method = 'rep'

# override filepath via args
if len(sys.argv) >= 3:
    mat_file_name = sys.argv[1]
    split_method = sys.argv[2]
if len(sys.argv) >= 4:
    root_path = sys.argv[3]

mat_path = root_path + mat_file_name
digits = io.loadmat(mat_path)
print("parameter, file path: ", mat_path)
item_count, h, w = digits.get('feature_matrix').shape[:3]
c = digits.get('feature_matrix').shape[3] if len(digits.get('feature_matrix').shape) > 3 else 1
h, w, kernel_size, kernel_stride, pool_stride, pool_size_list = model_parameter.select_parameter(h, w)
input_shape = (h, w, c)
print("input_shape: ", input_shape)
if h * w > 30000:
    mini_batch_size //= (1 + (h * w // 30000))
    print('mini batch size adjusted to: ', mini_batch_size)

cnn_model = get_cnn_model()

# redirect output to both console and txt
sys.stdout = model_parameter.Logger(
    root_path + '/log_train_' + mat_file_name.split('.')[0] + '_' + split_method + '.log')

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

x_train, x_val, y_train, y_val = model_selection.train_test_split(train_data, train_label, test_size=0.2,
                                                                  random_state=42)
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
cnn_model.compile(loss=keras.losses.categorical_crossentropy,
                  # optimizer=keras.optimizers.SGD(lr=0.01),
                  optimizer=optimizers.RMSprop(lr=learning_rate, decay=1e-5),
                  metrics=['accuracy'])
model_save_path = root_path + '/model_' + mat_file_name.split('.')[0] + '_' + split_method + '.h5'
monitor_criteria = 'val_loss'
checkpoint = ModelCheckpoint(model_save_path, monitor=monitor_criteria, verbose=1,
                             save_best_only=True, save_weights_only=False, mode='auto', period=1)

earlystop = EarlyStopping(monitor=monitor_criteria, min_delta=0, patience=60, verbose=0, mode='auto',
                          restore_best_weights=True)
callback_list = [checkpoint, earlystop]
cnn_model.summary()
history = cnn_model.fit(x_train, y_train,
                        batch_size=mini_batch_size,
                        epochs=n_epoch,
                        verbose=2,
                        validation_data=(x_val, y_val),
                        callbacks=callback_list)
# model.save_weights(root_path + '/weight_' + mat_file_name.split('.')[0] + '_' + split_method + '.h5')
# model.save(model_save_path)
score = cnn_model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Plot training & validation accuracy values
# fig, ax1 = plt.subplots()
# ax2 = ax1.twinx()
# acc = ax1.plot(history.history['acc'], label='Train acc')
# val_acc = ax1.plot(history.history['val_acc'], label='Val acc')
# loss = ax2.plot(history.history['loss'], label='Train loss')
# val_loss = ax2.plot(history.history['val_loss'], label='Val loss')
# lines = acc + val_acc + loss + val_loss
# labs = [l.get_label() for l in lines]
# ax1.legend(lines, labs, loc=0)
# plt.title('Model accuracy')
# ax1.set_ylabel('Accuracy')
# ax2.set_ylabel('Loss')
# plt.xlabel('Epoch')
# plt.show()

# intensity test
emotion_list = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
for i in range(len(normal_test_sets)):
    test_set = np.array(normal_test_sets[i])
    y_test = keras.utils.to_categorical(np.array([i] * len(test_set)), category_count)
    score = cnn_model.evaluate(test_set, y_test, verbose=0)
    print('{} Test accuracy:\t{}'.format(emotion_list[i], score[1]))

for i in range(len(strong_test_sets) - 1):
    i += 1
    test_set = np.array(strong_test_sets[i])
    y_test = keras.utils.to_categorical(np.array([i] * len(test_set)), category_count)
    score = cnn_model.evaluate(test_set, y_test, verbose=0)
    print('{} Test accuracy:\t{}'.format(emotion_list[i], score[1]))
