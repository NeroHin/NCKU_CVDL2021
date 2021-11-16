import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import load_img

# hyperparameters info
batch_size = 16
learning_rate = 0.0001
epochs = 100
num_classes = 10
subtract_pixel_mean = True
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

# cifar10 laebl dict
label_dict={0:"airplain",1:"automobile",2:"bird",3:"cat",4:"deer",5:"dog",
            6:"frog",7:"horse",8:"ship",9:"truck"} 

optimizer = optimizers.gradient_descent_v2.SGD(learning_rate=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)

# load cifar10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)


# Input image dimensions.
input_shape = x_train.shape[1:]

# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# If subtract pixel mean is enabled
if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

# 5.1 Load Cifar10 training dataset, and then show  9 Images(Pop-up) and Labels respectively (4%)
def show_Cifar10_training_img():

	for i in range(9):

		# define subplot
		plt.subplot(330 + 1 + i)

		# close axis
		plt.axis("off")

		# plot raw pixel data
		plt.imshow(x_train[i], cmap=plt.get_cmap('gray'))

		# show label title
		title = label_dict[y_train[i][0]]
		plt.title (title, fontsize=10) 
	plt.show()

    
# 5.2 Print out training hyperparameters on the terminal (batch size, learning rate, optimizer). (4%)


def print_training_hyperparameters_info():
    print('########################################')
    print('Batch size: {}'.format(batch_size))
    print('Optimizer : {}'.format(optimizer.get_config()))
    print('########################################')


# 5.3 Construct and show your model structure by print out on the terminal  (4%)

def print_model_structe():

    global model 

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                    input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.summary()
    return model

# 5.4 Training your model , than save your model and take a screenshot of your training loss and accuracy, and show the result.

def show_auc_and_loss_img():

    plt.figure(figsize=(19,28))
    plt.subplot(121)
    auc = mpimg.imread('/Users/nerohin/NCKU_CVDL2021/homework_1/Q5.Training_Cifar10_Classifier_Using_VGG16/vgg16_accuracy.png')
    plt.imshow(auc)
    plt.axis("off")

    plt.subplot(122)
    loss = mpimg.imread('/Users/nerohin/NCKU_CVDL2021/homework_1/Q5.Training_Cifar10_Classifier_Using_VGG16/vgg16_loss.png')
    plt.imshow(loss)
    plt.axis("off")
    plt.show()


# 5.6 tarin your model

def tarin_model():

    print_model_structe()
    model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(x_test, y_test),
                shuffle=True)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)


# Main function
if __name__ == '__main__':

    show_Cifar10_training_img()
    print_training_hyperparameters_info()
    print_model_structe()
    show_auc_and_loss_img()
    show_Cifar10_training_img()
    tarin_model()


