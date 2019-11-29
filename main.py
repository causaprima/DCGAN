import os
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from keras.layers import Conv2D, LeakyReLU, BatchNormalization, Dense, AveragePooling2D, GaussianNoise, MaxPooling2D
from keras.layers import Reshape, UpSampling2D, Activation, Dropout, Flatten, Conv2DTranspose, Input
from keras.models import model_from_json, Sequential, Model
from keras.optimizers import Adam, SGD
from sklearn.model_selection import train_test_split
import tensorflow as tf

DATA_PATH = '/Users/home/PycharmProjects/GAN/data/512x512'
RESULT_PATH = '/Users/home/PycharmProjects/GAN/result'
MODELS_PATH = '/Users/home/PycharmProjects/GAN/models'
TEST_RESULT_PATH = '/Users/home/PycharmProjects/GAN/test_result'
LOSS_PATH = '/Users/home/PycharmProjects/GAN'

# Google Colab paths
# DATA_PATH = '/content/drive/My Drive/DCGAN/Stenosis/data'
# RESULT_PATH = '/content/drive/My Drive/DCGAN/Stenosis/result'
# MODELS_PATH = '/content/drive/My Drive/DCGAN/Stenosis/models'
# TEST_RESULT_PATH = '/content/drive/My Drive/DCGAN/Stenosis/test_result'
# LOSS_PATH = '/content/drive/My Drive/DCGAN/Stenosis'

MODE = 'train'      # train, test, retrain
LAST_MODEL_NUM = ''

NUM_TO_RETRAIN = None

# IMAGE INFO
IMG_WIDTH = 512
IMG_HEIGHT = 512
CHANNELS = 1
IMG_SHAPE = (IMG_WIDTH, IMG_HEIGHT, CHANNELS)

# MODEL INFO
BATCH_SIZE = 6
EPOCHS = 10001
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 1 - TRAIN_RATIO - VAL_RATIO
LATENT_DIM = 100

CONV_ACTIVATION_FUNC = LeakyReLU
DECONV_ACTIVATION_FUNC = 'relu'
LEARNING_RATE = 0.002
OPTIMIZER = Adam(0.0002, 0.5)
LOSS = 'binary_crossentropy'


class DCGAN(object):
    def __init__(self, mode=MODE, img_width=IMG_WIDTH, img_height=IMG_HEIGHT, channels=CHANNELS, img_shape=IMG_SHAPE,
                 bath_size=BATCH_SIZE, epochs=EPOCHS, conv_activation_func=CONV_ACTIVATION_FUNC,
                 deconv_activation_func=DECONV_ACTIVATION_FUNC, learning_rate=LEARNING_RATE, optimizer=OPTIMIZER,
                 loss=LOSS, latent_dim=LATENT_DIM):
        self.mode = mode
        self.img_width = img_width
        self.img_height = img_height
        self.channels = channels
        self.img_shape = img_shape
        self.batch_size = bath_size
        self.epochs = epochs
        self.conv_activation_func = conv_activation_func
        self.deconv_activation_func = deconv_activation_func
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.loss = loss
        self.latent_dim = latent_dim

        self.discriminator_losses = []
        self.generator_losses = []

        self.start_epoch = 0

    # ----------------------------------------------------------------------------------------------------------------------
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])

        self.generator = self.build_generator()

        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        self.discriminator.trainable = False

        valid = self.discriminator(img)

        self.combined = Model(z, valid)
        self.combined.compile(loss=self.loss, optimizer=self.optimizer)

# ----------------------------------------------------------------------------------------------------------------------
    def convolution_block(self, input, filters, kernel_size, padding):
        conv = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding)(input)
        act = self.conv_activation_func(alpha=0.2)(conv)
        drop = Dropout(0.25)(act)
        max_pool = MaxPooling2D()(drop)

        return max_pool

# ----------------------------------------------------------------------------------------------------------------------
    def deconvolution_block(self, input, filters, kernel_size, padding='same'):
        conv = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding)(input)
        norm = BatchNormalization()(conv)
        act = Activation(self.deconv_activation_func)(norm)
        up_sampling = UpSampling2D()(act)

        return up_sampling

# ----------------------------------------------------------------------------------------------------------------------
    def build_discriminator(self):
        discriminator_input = Input(shape=self.img_shape)
        discriminator_data = discriminator_input
        for i in range(1, 10):
            discriminator_data = self.convolution_block(discriminator_data, 2**i, 3, 'same')
        discriminator_flatten = Flatten()(discriminator_data)
        discriminator_output = Dense(1, activation='sigmoid')(discriminator_flatten)

        model = Model(discriminator_input, discriminator_output, name='discriminator')
        model.summary()

        return model

# ----------------------------------------------------------------------------------------------------------------------
    def build_generator(self):
        generator_input = Input(shape=(self.latent_dim,))
        generator_data = Dense(self.img_width*2*2, activation=self.deconv_activation_func)(generator_input)
        generator_data = Reshape((2, 2, self.img_width))(generator_data)
        for i in range(9, 1, -1):
            generator_data = self.deconvolution_block(generator_data, 2**i, 3, 'same')
        generator_conv = Conv2D(filters=self.channels, kernel_size=3, padding='same')(generator_data)
        generator_output = Activation('sigmoid')(generator_conv)

        model = Model(generator_input, generator_output, name='generator')
        model.summary()

        return model

# ----------------------------------------------------------------------------------------------------------------------
    def train(self):
        X_train = train_images

        valid = np.ones((self.batch_size, 1))
        fake = np.zeros((self.batch_size, 1))

        for epoch in range(self.start_epoch, self.epochs):

            order = np.random.permutation(len(X_train))
            for start_index in range(0, len(X_train), self.batch_size):
                batch_indexes = order[start_index:start_index + self.batch_size]
                X_batch = X_train[batch_indexes]

                noise = np.random.uniform(-1, 1, size=[self.batch_size, self.latent_dim])
                gen_imgs = self.generator.predict(noise)

                # ---------------------
                #  Train Discriminator
                # ---------------------

                d_loss_real = self.discriminator.train_on_batch(X_batch, valid)
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # ---------------------
                #  Train Generator
                # ---------------------

                g_loss = self.combined.train_on_batch(noise, valid)

                # Plot the progress

            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            if epoch % 100 == 0:
                self.discriminator_losses.append(d_loss[0] + d_loss[1])
                self.generator_losses.append(g_loss)
                self.save_imgs(epoch)

            if epoch % 1000 == 0:
                self.save_models_and_weights(epoch)
                self.plot_loss()

# ----------------------------------------------------------------------------------------------------------------------
    def save_imgs(self, epoch):
        x_num, y_num = 3, 3
        noise = np.random.uniform(-1.0, 1.0, size=[x_num * y_num, self.latent_dim])
        gen_imgs = self.generator.predict(noise)

        # Rescale images -1;1 -> 0;1
        gen_imgs = (gen_imgs + 1) / 2.0

        gen_imgs = np.concatenate([np.concatenate(gen_imgs[x_num * (i-1):x_num * i],
                                                  axis=1) for i in range(1, y_num + 1)], axis=0)

        gen_imgs = np.uint8(gen_imgs * 255)
        gen_imgs = gen_imgs.reshape(gen_imgs.shape[0], gen_imgs.shape[1])

        plt.imsave(RESULT_PATH + "/i_%d.png" % epoch, gen_imgs, cmap='gray')

# ----------------------------------------------------------------------------------------------------------------------
    def test(self, num):
        for i in range(num):
            noise = np.random.uniform(-1.0, 1.0, size=[1, self.latent_dim])
            gen_imgs = self.generator.predict(noise)
            gen_imgs = (gen_imgs + 1) / 2.0
            gen_imgs = np.uint8(gen_imgs * 255)
            gen_imgs = gen_imgs.reshape(512, 512)

            plt.imsave(RESULT_PATH + '/test_result/' + str(i) + '.png', gen_imgs, cmap='gray')

# ----------------------------------------------------------------------------------------------------------------------
    def save_models_and_weights(self, num):
        gen_json = self.generator.to_json()
        dis_json = self.discriminator.to_json()

        with open(MODELS_PATH + "/gen.json", "w+") as json_file:
            json_file.write(gen_json)

        with open(MODELS_PATH + "/dis.json", "w+") as json_file:
            json_file.write(dis_json)

        self.generator.save_weights(MODELS_PATH + "/weights/gen" + str(num) + ".h5")
        self.discriminator.save_weights(MODELS_PATH + "/weights/dis" + str(num) + ".h5")

        print(f"Model number {str(num)} saved!")

# ----------------------------------------------------------------------------------------------------------------------
    def load_models_and_weight(self, num):
        self.start_epoch = num
        self.epochs = EPOCHS + num

        # Generator
        gen_file = open(MODELS_PATH + "/gen.json", 'r')
        gen_json = gen_file.read()
        gen_file.close()

        self.generator = model_from_json(gen_json)
        self.generator.load_weights(MODELS_PATH + "/weights/gen" + str(num) + ".h5")

        # Discriminator
        dis_file = open(MODELS_PATH + "/dis.json", 'r')
        dis_json = dis_file.read()
        dis_file.close()

        self.discriminator = model_from_json(dis_json)
        self.discriminator.load_weights(MODELS_PATH + "/weights/dis" + str(num) + ".h5")

        # Reinitialize
        self.discriminator.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])

        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        self.discriminator.trainable = False

        valid = self.discriminator(img)

        self.combined = Model(z, valid)
        self.combined.compile(loss=self.loss, optimizer=self.optimizer)

# ----------------------------------------------------------------------------------------------------------------------
    def plot_loss(self):
        plt.figure(figsize=(10, 5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(self.generator_losses, label="G")
        plt.plot(self.discriminator_losses, label="D")
        plt.xlabel("epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(LOSS_PATH + '/loss.png')

# ----------------------------------------------------------------------------------------------------------------------

def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH), cv2.INTER_CUBIC)
    img = img / 255.0
    img = np.clip(img, a_min=0, a_max=1)
    img = np.expand_dims(img, 2)
    return img

all_img_paths = glob.glob(DATA_PATH + "/*.bmp")
train_img_paths, test_img_paths, = train_test_split(all_img_paths,
                                                    test_size=TEST_RATIO,
                                                    random_state=11)
train_img_paths,  val_img_paths = train_test_split(train_img_paths,
                                                   test_size=VAL_RATIO/(TRAIN_RATIO+VAL_RATIO),
                                                   random_state=11)

train_images = np.array([np.array(preprocess_image(fname)) for fname in train_img_paths])
val_images = np.array([np.array(preprocess_image(fname)) for fname in val_img_paths])
test_images = np.array([np.array(preprocess_image(fname)) for fname in test_img_paths])

# ----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    dcgan = DCGAN()
    if MODE == 'train':
        dcgan.train()
    elif MODE == 'test':
        dcgan.load_models_and_weight(LAST_MODEL_NUM)
        dcgan.test(10)
    elif MODE == 'retrain':
        dcgan.load_models_and_weight(LAST_MODEL_NUM)
        dcgan.train()
