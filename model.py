import keras.backend as K
import tensorflow as tf
from keras.layers import Input, Conv2D, BatchNormalization, UpSampling2D
from keras.models import Model
from keras.utils import multi_gpu_model
from keras.utils import plot_model

from config import img_rows, img_cols, num_classes


def build_encoder_decoder():
    kernel = 3

    input_tensor = Input(shape=(img_rows, img_cols, 1))
    x = Conv2D(64, (kernel, kernel), activation='relu', padding='same', name='conv1_1')(input_tensor)
    x = Conv2D(64, (kernel, kernel), activation='relu', padding='same', name='conv1_2', strides=(2, 2))(x)
    x = BatchNormalization()(x)

    x = Conv2D(128, (kernel, kernel), activation='relu', padding='same', name='conv2_1')(x)
    x = Conv2D(128, (kernel, kernel), activation='relu', padding='same', name='conv2_2', strides=(2, 2))(x)
    x = BatchNormalization()(x)

    x = Conv2D(256, (kernel, kernel), activation='relu', padding='same', name='conv3_1')(x)
    x = Conv2D(256, (kernel, kernel), activation='relu', padding='same', name='conv3_2')(x)
    x = Conv2D(256, (kernel, kernel), activation='relu', padding='same', name='conv3_3', strides=(2, 2))(x)
    x = BatchNormalization()(x)

    x = Conv2D(512, (kernel, kernel), activation='relu', padding='same', name='conv4_1')(x)
    x = Conv2D(512, (kernel, kernel), activation='relu', padding='same', name='conv4_2')(x)
    x = Conv2D(512, (kernel, kernel), activation='relu', padding='same', name='conv4_3')(x)
    x = BatchNormalization()(x)

    x = Conv2D(512, (kernel, kernel), activation='relu', padding='same', dilation_rate=2, name='conv5_1')(x)
    x = Conv2D(512, (kernel, kernel), activation='relu', padding='same', dilation_rate=2, name='conv5_2')(x)
    x = Conv2D(512, (kernel, kernel), activation='relu', padding='same', dilation_rate=2, name='conv5_3')(x)
    x = BatchNormalization()(x)

    x = Conv2D(512, (kernel, kernel), activation='relu', padding='same', dilation_rate=2, name='conv6_1')(x)
    x = Conv2D(512, (kernel, kernel), activation='relu', padding='same', dilation_rate=2, name='conv6_2')(x)
    x = Conv2D(512, (kernel, kernel), activation='relu', padding='same', dilation_rate=2, name='conv6_3')(x)
    x = BatchNormalization()(x)

    x = Conv2D(256, (kernel, kernel), activation='relu', padding='same', name='conv7_1')(x)
    x = Conv2D(256, (kernel, kernel), activation='relu', padding='same', name='conv7_2')(x)
    x = Conv2D(256, (kernel, kernel), activation='relu', padding='same', name='conv7_3')(x)
    x = BatchNormalization()(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(128, (kernel, kernel), activation='relu', padding='same', name='conv8_1')(x)
    x = Conv2D(128, (kernel, kernel), activation='relu', padding='same', name='conv8_2')(x)
    x = Conv2D(128, (kernel, kernel), activation='relu', padding='same', name='conv8_3')(x)
    x = BatchNormalization()(x)

    outputs = Conv2D(num_classes, (1, 1), activation='softmax', padding='same', name='pred')(x)

    model = Model(inputs=input_tensor, outputs=outputs, name="ColorNet")
    return model


if __name__ == '__main__':
    with tf.device("/cpu:0"):
        encoder_decoder = build_encoder_decoder()
    print(encoder_decoder.summary())
    plot_model(encoder_decoder, to_file='encoder_decoder.svg', show_layer_names=True, show_shapes=True)

    parallel_model = multi_gpu_model(encoder_decoder, gpus=None)
    print(parallel_model.summary())
    plot_model(parallel_model, to_file='parallel_model.svg', show_layer_names=True, show_shapes=True)

    K.clear_session()
