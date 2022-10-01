from logging import config
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from einops import rearrange
import segmentation_models as sm
from tensorflow.keras.models import Model
import keras_unet_collection.models as kuc
from tensorflow.keras import backend as K
from tensorflow.keras.layers import LeakyReLU, add, Conv2D, PReLU, ReLU, Concatenate, Activation, MaxPool2D, Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda




def ad_unet(config):
    """
        Summary:
            Create dynamic MNET model object based on input shape
        Arguments: 
            Model configuration from config.yaml
        Return:
            Keras.model object
    """
    no_layer = 0
    inp_size = config["height"]
    start_filter = 4
    while inp_size>=8:
        no_layer += 1
        inp_size = inp_size / 2
    
    # building model encoder
    encoder = {}
    inputs = Input((config['height'], config['width'], config['in_channels']))
    for i in range(no_layer):
        if i == 0:
            encoder["enc_{}_0".format(i)] = Conv2D(start_filter, (3, 3), name = "enc_{}_0".format(i),activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
        else:
            encoder["enc_{}_0".format(i)] = Conv2D(start_filter, (3, 3), name = "enc_{}_0".format(i), activation='relu', kernel_initializer='he_normal', padding='same')(encoder["mp_{}".format(i-1)])
        start_filter *= 2
        encoder["enc_{}_1".format(i)] = Conv2D(start_filter, (3, 3), name = "enc_{}_1".format(i), activation='relu', kernel_initializer='he_normal', padding='same')(encoder["enc_{}_0".format(i)])
        encoder["mp_{}".format(i)] = MaxPooling2D((2,2), name = "mp_{}".format(i))(encoder["enc_{}_1".format(i)])
    
    # building model middle layer
    mid_1 = Conv2D(start_filter, (3, 3), name = "mid_1", activation='relu', kernel_initializer='he_normal', padding='same')(encoder["mp_{}".format(no_layer-1)])
    start_filter *= 2
    mid_drop = Dropout(0.3)(mid_1)
    mid_2 = Conv2D(start_filter, (3, 3), name = "mid_2", activation='relu', kernel_initializer='he_normal', padding='same')(mid_drop)
    
    
    # building model decoder
    start_filter = start_filter / 2
    decoder = {}
    for i in range(no_layer):
        if i == 0:
            decoder["dec_T_{}".format(i)] = Conv2DTranspose(start_filter, (2, 2), name = "dec_T_{}".format(i), strides=(2, 2), padding='same')(mid_2)
        else:
            decoder["dec_T_{}".format(i)] = Conv2DTranspose(start_filter, (2, 2), name = "dec_T_{}".format(i), strides=(2, 2), padding='same')(decoder["dec_{}_1".format(i-1)])
        decoder["cc_{}".format(i)] = concatenate([decoder["dec_T_{}".format(i)], encoder["enc_{}_1".format(no_layer-i-1)]], axis=3)
        start_filter = start_filter / 2
        decoder["dec_{}_0".format(i)] = Conv2D(start_filter, (3, 3), name = "dec_{}_0".format(i), activation='relu', kernel_initializer='he_normal', padding='same')(decoder["cc_{}".format(i)])
        decoder["dec_{}_1".format(i)] = Conv2D(start_filter, (3, 3), name = "dec_{}_1".format(i), activation='relu', kernel_initializer='he_normal', padding='same')(decoder["dec_{}_0".format(i)])
        
    
    # building output layer
    outputs = Conv2D(config['num_classes'], (1, 1), activation='softmax', dtype='float32')(decoder["dec_{}_1".format(no_layer-1)])
    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model



class CustomModel(keras.Model):
    #@tf.function(jit_compile=True)
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


# UNET Model
# ----------------------------------------------------------------------------------------------

def unet(config):
    
    """
        Summary:
            Create UNET model object
        Arguments: 
            Model configuration from config.yaml
        Return:
            Keras.model object
    """
    
    inputs = Input((config['height'], config['width'], config['in_channels']))
 
    #Contraction path
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = Dropout(0.2)(c1)  # Original 0.1
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.2)(c2)  # Original 0.1
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
     
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
     
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
     
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    
    #Expansive path 
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
     
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
     
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.2)(c8)  # Original 0.1
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
     
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.2)(c9)  # Original 0.1
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
     
    outputs = Conv2D(config['num_classes'], (1, 1), activation='softmax', dtype='float32')(c9)
     
    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model
    
    
    
# Modification UNET Model
# ----------------------------------------------------------------------------------------------


def mod_unet(config):
    
    """
        Summary:
            Create MNET model object
        Arguments: 
            Model configuration from config.yaml
        Return:
            Keras.model object
    """
    
    inputs = Input((config['height'], config['width'], config['in_channels']))
    
    #Contraction path
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = Dropout(0.2)(c1)  # Original 0.1
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.2)(c2)  # Original 0.1
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
     
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
     
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.2)(c5)  # Original 0.1
    c5 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    p5 = MaxPooling2D((2, 2))(c5)
     
    c6 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p5)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
    p6 = MaxPooling2D((2, 2))(c6)
     
    c7 = Conv2D(1012, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p6)
    c7 = Dropout(0.3)(c7)
    c7 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
    
    # Expansive path 
    
    u8 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c6])
    c8 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.2)(c8)
    c8 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
    
    u9 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c5])
    c9 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.2)(c9)
    c9 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
        
    u10 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c9)
    u10 = concatenate([u10, c4])
    c10 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u10)
    c10 = Dropout(0.2)(c10)
    c10 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c10)
     
    u11 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c10)
    u11 = concatenate([u11, c3])
    c11 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u11)
    c11 = Dropout(0.2)(c11)
    c11 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c11)
     
    u12 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c11)
    u12 = concatenate([u12, c2])
    c12 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u12)
    c12 = Dropout(0.2)(c12)  # Original 0.1
    c12 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c12)
     
    u13 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c12)
    u13 = concatenate([u13, c1], axis=3)
    c13 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u13)
    c13 = Dropout(0.2)(c13)  # Original 0.1
    c13 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c13)
     
    outputs = Conv2D(config['num_classes'], (1, 1), activation='softmax', dtype='float32')(c13)
     
    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model



# U2Net Model
# ----------------------------------------------------------------------------------------------

def basicblocks(input, filter, dilates = 1):
    x1 = Conv2D(filter, (3, 3), padding = 'same', dilation_rate = 1*dilates)(input)
    x1 = ReLU()(BatchNormalization()(x1))
    return x1

def RSU7(input, in_ch = 3, mid_ch = 12, out_ch = 3):
    hx = input
    #1
    hxin = basicblocks(hx, out_ch, 1)
    hx1 = basicblocks(hxin, mid_ch, 1)
    hx = keras.layers.MaxPool2D((2, 2), strides = 2)(hx1)
    #2
    hx2 = basicblocks(hx, mid_ch, 1)
    hx = keras.layers.MaxPool2D((2, 2), strides = 2)(hx2)
    #3
    hx3 = basicblocks(hx, mid_ch, 1)
    hx = keras.layers.MaxPool2D((2, 2), strides = 2)(hx3)
    #4
    hx4 = basicblocks(hx, mid_ch, 1)
    hx = keras.layers.MaxPool2D((2, 2), strides = 2)(hx4)
    #5
    hx5 = basicblocks(hx, mid_ch, 1)
    hx = keras.layers.MaxPool2D((2, 2), strides = 2)(hx5)
    #6
    hx6 = basicblocks(hx, mid_ch, 1)
    #7
    hx7 = basicblocks(hx6, mid_ch, 2)

    #down
    #6
    hx6d = Concatenate(axis = -1)([hx7, hx6])
    hx6d = basicblocks(hx6d, mid_ch, 1)
    a,b,c,d = K.int_shape(hx5)
    hx6d=keras.layers.UpSampling2D(size=(2,2))(hx6d)

    #5
    hx5d = Concatenate(axis=-1)([hx6d, hx5])
    hx5d = basicblocks(hx5d, mid_ch, 1)
    a, b, c, d = K.int_shape(hx4)
    hx5d = keras.layers.UpSampling2D(size=(2,2))(hx5d)

    # 4
    hx4d = Concatenate(axis=-1)([hx5d, hx4])
    hx4d = basicblocks(hx4d, mid_ch, 1)
    a, b, c, d = K.int_shape(hx3)
    hx4d = keras.layers.UpSampling2D(size=(2,2))(hx4d)

    # 3
    hx3d = Concatenate(axis=-1)([hx4d, hx3])
    hx3d = basicblocks(hx3d, mid_ch, 1)
    a, b, c, d = K.int_shape(hx2)
    hx3d = keras.layers.UpSampling2D(size=(2,2))(hx3d)

    # 2
    hx2d = Concatenate(axis=-1)([hx3d, hx2])
    hx2d = basicblocks(hx2d, mid_ch, 1)
    a, b, c, d = K.int_shape(hx1)
    hx2d = keras.layers.UpSampling2D(size=(2,2))(hx2d)

    # 1
    hx1d = Concatenate(axis=-1)([hx2d, hx1])
    hx1d = basicblocks(hx1d, out_ch, 1)

    #output
    output=keras.layers.add([hx1d,hxin])
    return output

def RSU6(input,in_ch=3,mid_ch=12,out_ch=3):
    hx=input
    #1
    hxin=basicblocks(hx,out_ch,1)
    hx1=basicblocks(hxin,mid_ch,1)
    hx=keras.layers.MaxPool2D((2,2),strides=2)(hx1)
    #2
    hx2=basicblocks(hx,mid_ch,1)
    hx=keras.layers.MaxPool2D((2,2),strides=2)(hx2)
    #3
    hx3 = basicblocks(hx, mid_ch, 1)
    hx = keras.layers.MaxPool2D((2, 2), strides=2)(hx3)
    #4
    hx4=basicblocks(hx,mid_ch,1)
    hx=keras.layers.MaxPool2D((2,2),strides=2)(hx4)
    #5
    hx5 = basicblocks(hx, mid_ch, 1)
    hx = keras.layers.MaxPool2D((2, 2), strides=2)(hx5)
    #6
    hx6=basicblocks(hx,mid_ch,1)
    hx6=keras.layers.UpSampling2D((2, 2))(hx6)

    #5
    hx5d = Concatenate(axis=-1)([hx6, hx5])
    hx5d = basicblocks(hx5d, mid_ch, 1)
    a, b, c, d = K.int_shape(hx4)
    hx5d = keras.layers.UpSampling2D(size=(2,2))(hx5d)

    # 4
    hx4d = Concatenate(axis=-1)([hx5d, hx4])
    hx4d = basicblocks(hx4d, mid_ch, 1)
    a, b, c, d = K.int_shape(hx3)
    hx4d = keras.layers.UpSampling2D(size=(2,2))(hx4d)

    # 3
    hx3d = Concatenate(axis=-1)([hx4d, hx3])
    hx3d = basicblocks(hx3d, mid_ch, 1)
    a, b, c, d = K.int_shape(hx2)
    hx3d = keras.layers.UpSampling2D(size=(2,2))(hx3d)

    # 2
    hx2d = Concatenate(axis=-1)([hx3d, hx2])
    hx2d = basicblocks(hx2d, mid_ch, 1)
    a, b, c, d = K.int_shape(hx1)
    hx2d = keras.layers.UpSampling2D(size=(2,2))(hx2d)

    # 1
    hx1d = Concatenate(axis=-1)([hx2d, hx1])
    hx1d = basicblocks(hx1d, out_ch, 1)

    #output
    output=keras.layers.add([hx1d,hxin])
    return output

def RSU5(input,in_ch=3,mid_ch=12,out_ch=3):
    hx=input
    #1
    hxin=basicblocks(hx,out_ch,1)
    hx1=basicblocks(hxin,mid_ch,1)
    hx=keras.layers.MaxPool2D((2,2),strides=2)(hx1)
    #2
    hx2=basicblocks(hx,mid_ch,1)
    hx=keras.layers.MaxPool2D((2,2),strides=2)(hx2)
    #3
    hx3 = basicblocks(hx, mid_ch, 1)
    hx = keras.layers.MaxPool2D((2, 2), strides=2)(hx3)
    #4
    hx4=basicblocks(hx,mid_ch,1)
    hx=keras.layers.MaxPool2D((2,2),strides=2)(hx4)
    #5
    hx5 = basicblocks(hx, mid_ch, 1)
    #hx5 = keras.layers.MaxPool2D((2, 2), strides=2)(hx5)
    hx5 = keras.layers.UpSampling2D((2, 2))(hx5)
    # 4
    hx4d = Concatenate(axis=-1)([hx5, hx4])
    hx4d = basicblocks(hx4d, mid_ch, 1)
    a, b, c, d = K.int_shape(hx3)
    hx4d = keras.layers.UpSampling2D(size=(2,2))(hx4d)

    # 3
    hx3d = Concatenate(axis=-1)([hx4d, hx3])
    hx3d = basicblocks(hx3d, mid_ch, 1)
    a, b, c, d = K.int_shape(hx2)
    hx3d = keras.layers.UpSampling2D(size=(2,2))(hx3d)

    # 2
    hx2d = Concatenate(axis=-1)([hx3d, hx2])
    hx2d = basicblocks(hx2d, mid_ch, 1)
    a, b, c, d = K.int_shape(hx1)
    hx2d = keras.layers.UpSampling2D(size=(2,2))(hx2d)

    # 1
    hx1d = Concatenate(axis=-1)([hx2d, hx1])
    hx1d = basicblocks(hx1d, out_ch, 1)

    #output
    output=keras.layers.add([hx1d,hxin])
    return output

def RSU4(input,in_ch=3,mid_ch=12,out_ch=3):
    hx=input
    #1
    hxin=basicblocks(hx,out_ch,1)
    hx1=basicblocks(hxin,mid_ch,1)
    hx=keras.layers.MaxPool2D((2,2),strides=2)(hx1)
    #2
    hx2=basicblocks(hx,mid_ch,1)
    hx=keras.layers.MaxPool2D((2,2),strides=2)(hx2)
    #3
    hx3 = basicblocks(hx, mid_ch, 1)
    hx = keras.layers.MaxPool2D((2, 2), strides=2)(hx3)
    #4
    hx4=basicblocks(hx,mid_ch,1)
    hx4=keras.layers.UpSampling2D((2,2))(hx4)

    # 3
    hx3d = Concatenate(axis=-1)([hx4, hx3])
    hx3d = basicblocks(hx3d, mid_ch, 1)
    a, b, c, d = K.int_shape(hx2)
    hx3d = keras.layers.UpSampling2D(size=(2,2))(hx3d)

    # 2
    hx2d = Concatenate(axis=-1)([hx3d, hx2])
    hx2d = basicblocks(hx2d, mid_ch, 1)
    a, b, c, d = K.int_shape(hx1)
    hx2d = keras.layers.UpSampling2D(size=(2,2))(hx2d)

    # 1
    hx1d = Concatenate(axis=-1)([hx2d, hx1])
    hx1d = basicblocks(hx1d, out_ch, 1)

    #output
    output=keras.layers.add([hx1d,hxin])
    return output

def RSU4f(input, in_ch = 3, mid_ch = 12, out_ch = 3):
    hx=input
    #1
    hxin = basicblocks(hx, out_ch, 1)
    hx1 = basicblocks(hxin, mid_ch, 1)
    #2
    hx2=basicblocks(hx, mid_ch, 2)
    #3
    hx3 = basicblocks(hx, mid_ch, 4)
    #4
    hx4=basicblocks(hx, mid_ch, 8)

    # 3
    hx3d = Concatenate(axis = -1)([hx4, hx3])
    hx3d = basicblocks(hx3d, mid_ch, 4)

    # 2
    hx2d = Concatenate(axis = -1)([hx3d, hx2])
    hx2d = basicblocks(hx2d, mid_ch, 2)

    # 1
    hx1d = Concatenate(axis = -1)([hx2d, hx1])
    hx1d = basicblocks(hx1d, out_ch, 1)

    #output
    output = keras.layers.add([hx1d, hxin])
    return output

def u2net(config):
    
    """
        Summary:
            Create U2NET model object
        Arguments: 
            Model configuration from config.yaml
        Return:
            Keras.model object
    """
    
    input = Input((config['height'], config['width'], config['in_channels']))

    stage1 = RSU7(input, in_ch = 3, mid_ch = 32, out_ch = 64)
    stage1p = keras.layers.MaxPool2D((2,2), strides = 2)(stage1)

    stage2 = RSU6(stage1p, in_ch = 64, mid_ch = 32, out_ch = 128)
    stage2p = keras.layers.MaxPool2D((2, 2), strides = 2)(stage2)

    stage3 = RSU5(stage2p, in_ch = 128, mid_ch = 64, out_ch = 256)
    stage3p = keras.layers.MaxPool2D((2, 2), strides = 2)(stage3)

    stage4 = RSU4(stage3p, in_ch = 256, mid_ch = 128, out_ch = 512)
    stage4p = keras.layers.MaxPool2D((2, 2), strides = 2)(stage4)

    stage5 = RSU4f(stage4p, in_ch = 512, mid_ch = 256, out_ch = 512)
    stage5p = keras.layers.MaxPool2D((2, 2), strides = 2)(stage5)

    stage6 = RSU4f(stage5, in_ch = 512, mid_ch = 256, out_ch = 512)
    stage6u = keras.layers.UpSampling2D((1, 1))(stage6)

    #decoder
    stage6a = Concatenate(axis = -1)([stage6u,stage5])
    stage5d = RSU4f(stage6a, 1024, 256, 512)
    stage5du = keras.layers.UpSampling2D((2, 2))(stage5d)

    stage5a = Concatenate(axis = -1)([stage5du, stage4])
    stage4d = RSU4(stage5a, 1024, 128, 256)
    stage4du = keras.layers.UpSampling2D((2, 2))(stage4d)

    stage4a = Concatenate(axis = -1)([stage4du, stage3])
    stage3d = RSU5(stage4a, 512, 64, 128)
    stage3du = keras.layers.UpSampling2D((2, 2))(stage3d)

    stage3a = Concatenate(axis = -1)([stage3du, stage2])
    stage2d = RSU6(stage3a, 256, 32, 64)
    stage2du = keras.layers.UpSampling2D((2, 2))(stage2d)

    stage2a = Concatenate(axis = -1)([stage2du, stage1])
    stage1d = RSU6(stage2a, 128, 16, 64)

    #side output
    side1 = Conv2D(config['num_classes'], (3, 3), padding = 'same', name = 'side1')(stage1d)
    side2 = Conv2D(config['num_classes'], (3, 3), padding = 'same')(stage2d)
    side2 = keras.layers.UpSampling2D((2, 2), name = 'side2')(side2)
    side3 = Conv2D(config['num_classes'], (3, 3), padding = 'same')(stage3d)
    side3 = keras.layers.UpSampling2D((4, 4), name = 'side3')(side3)
    side4 = Conv2D(config['num_classes'], (3, 3), padding = 'same')(stage4d)
    side4 = keras.layers.UpSampling2D((8, 8), name = 'side4')(side4)
    side5 = Conv2D(config['num_classes'], (3, 3), padding = 'same')(stage5d)
    side5 = keras.layers.UpSampling2D((16, 16), name = 'side5')(side5)
    side6 = Conv2D(config['num_classes'], (3, 3), padding = 'same')(stage6)
    side6 = keras.layers.UpSampling2D((16, 16), name = 'side6')(side6)

    out = Concatenate(axis = -1)([side1, side2, side3, side4, side5, side6])
    out = Conv2D(config['num_classes'], (1, 1), padding = 'same', name = 'out', dtype='float32')(out)

    # model = Model(inputs = [input], outputs = [side1, side2, side3, side4, side5, side6, out])
    model = Model(inputs = [input], outputs = [out])
    
    return model
    


# DnCNN Model
# ----------------------------------------------------------------------------------------------

def DnCNN(config):
    
    """
        Summary:
            Create DNCNN model object
        Arguments: 
            Model configuration from config.yaml
        Return:
            Keras.model object
    """
    
    inpt = Input(shape=(config['height'], config['width'], config['in_channels']))
    # 1st layer, Conv+relu
    x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(inpt)
    x = Activation('relu')(x)
    # 15 layers, Conv+BN+relu
    for i in range(15):
        x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(x)
        x = BatchNormalization(axis=-1, epsilon=1e-3)(x)
        x = Activation('relu')(x)   
    # last layer, Conv
    x = Conv2D(config['num_classes'], (1, 1), activation='softmax',dtype='float32')(x)
    # x = Conv2D(filters=6, kernel_size=(3,3), strides=(1,1), padding='same')(x)
    # x = tf.keras.layers.Subtract()([inpt, x])   # input - noise
    model = Model(inputs=inpt, outputs=x)
    
    return model


# 2D-VNET Model
# ----------------------------------------------------------------------------------------------

def resBlock(input, stage, keep_prob, stage_num = 5):
    
    for _ in range(3 if stage>3 else stage):
        conv = PReLU()(BatchNormalization()(Conv2D(16*(2**(stage-1)), 5, activation = None, padding = 'same', kernel_initializer = 'he_normal')(input)))
        # print('conv_down_stage_%d:' %stage,conv.get_shape().as_list())
    conv_add = PReLU()(add([input, conv]))
    # print('conv_add:',conv_add.get_shape().as_list())
    conv_drop = Dropout(keep_prob)(conv_add)
    
    if stage < stage_num:
        conv_downsample = PReLU()(BatchNormalization()(Conv2D(16*(2**stage), 2, strides=(2, 2),activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv_drop)))
        return conv_downsample, conv_add
    else:
        return conv_add, conv_add
    
def up_resBlock(forward_conv,input_conv,stage):
    
    conv = concatenate([forward_conv, input_conv], axis = -1)
    
    for _ in range(3 if stage>3 else stage):
        conv = PReLU()(BatchNormalization()(Conv2D(16*(2**(stage-1)), 5, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv)))
        conv_add = PReLU()(add([input_conv,conv]))

    if stage > 1:
        conv_upsample = PReLU()(BatchNormalization()(Conv2DTranspose(16*(2**(stage-2)),2,strides = (2, 2),padding = 'valid',activation = None,kernel_initializer = 'he_normal')(conv_add)))
        return conv_upsample
    else:
        return conv_add
    
def vnet(config):
    
    """
        Summary:
            Create VNET model object
        Arguments: 
            Model configuration from config.yaml
        Return:
            Keras.model object
    """
    
    keep_prob = 0.99
    features = []
    stage_num = 5 # number of blocks
    input = Input((config['height'], config['width'], config['in_channels']))
    x = PReLU()(BatchNormalization()(Conv2D(16, 5, activation = None, padding = 'same', kernel_initializer = 'he_normal')(input)))
    
    for s in range(1, stage_num+1):
        x, feature = resBlock(x, s, keep_prob, stage_num)
        features.append(feature)
        
    conv_up = PReLU()(BatchNormalization()(Conv2DTranspose(16*(2**(s-2)),2, strides = (2, 2), padding = 'valid', activation = None, kernel_initializer = 'he_normal')(x)))
    
    for d in range(stage_num-1, 0, -1):
        conv_up = up_resBlock(features[d-1], conv_up, d)

    output = Conv2D(config['num_classes'], 1, activation = 'softmax', padding = 'same', kernel_initializer = 'he_normal')(conv_up)
        
    model = Model(inputs = [input], outputs = [output])

    return model


# UNET++ Model
# ----------------------------------------------------------------------------------------------

def conv2d(filters: int):
    return Conv2D(filters = filters,
                  kernel_size = (3, 3),
                  padding='same')

def conv2dtranspose(filters: int):
    return Conv2DTranspose(filters = filters,
                           kernel_size = (2, 2),
                           strides = (2, 2),
                           padding = 'same')

def unet_plus_plus(config):
    
    """
        Summary:
            Create UNET++ model object
        Arguments: 
            Model configuration from config.yaml
        Return:
            Keras.model object
    """

    input = Input((config['height'], config['width'], config['in_channels']))

    x00 = conv2d(filters = int(16 * 2))(input)
    x00 = BatchNormalization()(x00)
    x00 = LeakyReLU(0.01)(x00)
    x00 = Dropout(0.2)(x00)
    x00 = conv2d(filters = int(16 * 2))(x00)
    x00 = BatchNormalization()(x00)
    x00 = LeakyReLU(0.01)(x00)
    x00 = Dropout(0.2)(x00)
    p0 = MaxPooling2D(pool_size=(2, 2))(x00)

    x10 = conv2d(filters = int(32 * 2))(p0)
    x10 = BatchNormalization()(x10)
    x10 = LeakyReLU(0.01)(x10)
    x10 = Dropout(0.2)(x10)
    x10 = conv2d(filters = int(32 * 2))(x10)
    x10 = BatchNormalization()(x10)
    x10 = LeakyReLU(0.01)(x10)
    x10 = Dropout(0.2)(x10)
    p1 = MaxPooling2D(pool_size=(2, 2))(x10)

    x01 = conv2dtranspose(int(16 * 2))(x10)
    x01 = concatenate([x00, x01])
    x01 = conv2d(filters = int(16 * 2))(x01)
    x01 = BatchNormalization()(x01)
    x01 = LeakyReLU(0.01)(x01)
    x01 = conv2d(filters = int(16 * 2))(x01)
    x01 = BatchNormalization()(x01)
    x01 = LeakyReLU(0.01)(x01)
    x01 = Dropout(0.2)(x01)

    x20 = conv2d(filters = int(64 * 2))(p1)
    x20 = BatchNormalization()(x20)
    x20 = LeakyReLU(0.01)(x20)
    x20 = Dropout(0.2)(x20)
    x20 = conv2d(filters = int(64 * 2))(x20)
    x20 = BatchNormalization()(x20)
    x20 = LeakyReLU(0.01)(x20)
    x20 = Dropout(0.2)(x20)
    p2 = MaxPooling2D(pool_size=(2, 2))(x20)

    x11 = conv2dtranspose(int(16 * 2))(x20)
    x11 = concatenate([x10, x11])
    x11 = conv2d(filters = int(16 * 2))(x11)
    x11 = BatchNormalization()(x11)
    x11 = LeakyReLU(0.01)(x11)
    x11 = conv2d(filters = int(16 * 2))(x11)
    x11 = BatchNormalization()(x11)
    x11 = LeakyReLU(0.01)(x11)
    x11 = Dropout(0.2)(x11)

    x02 = conv2dtranspose(int(16 * 2))(x11)
    x02 = concatenate([x00, x01, x02])
    x02 = conv2d(filters = int(16 * 2))(x02)
    x02 = BatchNormalization()(x02)
    x02 = LeakyReLU(0.01)(x02)
    x02 = conv2d(filters = int(16 * 2))(x02)
    x02 = BatchNormalization()(x02)
    x02 = LeakyReLU(0.01)(x02)
    x02 = Dropout(0.2)(x02)

    x30 = conv2d(filters = int(128 * 2))(p2)
    x30 = BatchNormalization()(x30)
    x30 = LeakyReLU(0.01)(x30)
    x30 = Dropout(0.2)(x30)
    x30 = conv2d(filters = int(128 * 2))(x30)
    x30 = BatchNormalization()(x30)
    x30 = LeakyReLU(0.01)(x30)
    x30 = Dropout(0.2)(x30)
    p3 = MaxPooling2D(pool_size=(2, 2))(x30)

    x21 = conv2dtranspose(int(16 * 2))(x30)
    x21 = concatenate([x20, x21])
    x21 = conv2d(filters = int(16 * 2))(x21)
    x21 = BatchNormalization()(x21)
    x21 = LeakyReLU(0.01)(x21)
    x21 = conv2d(filters = int(16 * 2))(x21)
    x21 = BatchNormalization()(x21)
    x21 = LeakyReLU(0.01)(x21)
    x21 = Dropout(0.2)(x21)

    x12 = conv2dtranspose(int(16 * 2))(x21)
    x12 = concatenate([x10, x11, x12])
    x12 = conv2d(filters = int(16 * 2))(x12)
    x12 = BatchNormalization()(x12)
    x12 = LeakyReLU(0.01)(x12)
    x12 = conv2d(filters = int(16 * 2))(x12)
    x12 = BatchNormalization()(x12)
    x12 = LeakyReLU(0.01)(x12)
    x12 = Dropout(0.2)(x12)

    x03 = conv2dtranspose(int(16 * 2))(x12)
    x03 = concatenate([x00, x01, x02, x03])
    x03 = conv2d(filters = int(16 * 2))(x03)
    x03 = BatchNormalization()(x03)
    x03 = LeakyReLU(0.01)(x03)
    x03 = conv2d(filters = int(16 * 2))(x03)
    x03 = BatchNormalization()(x03)
    x03 = LeakyReLU(0.01)(x03)
    x03 = Dropout(0.2)(x03)

    m = conv2d(filters = int(256 * 2))(p3)
    m = BatchNormalization()(m)
    m = LeakyReLU(0.01)(m)
    m = conv2d(filters = int(256 * 2))(m)
    m = BatchNormalization()(m)
    m = LeakyReLU(0.01)(m)
    m = Dropout(0.2)(m)

    x31 = conv2dtranspose(int(128 * 2))(m)
    x31 = concatenate([x31, x30])
    x31 = conv2d(filters = int(128 * 2))(x31)
    x31 = BatchNormalization()(x31)
    x31 = LeakyReLU(0.01)(x31)
    x31 = conv2d(filters = int(128 * 2))(x31)
    x31 = BatchNormalization()(x31)
    x31 = LeakyReLU(0.01)(x31)
    x31 = Dropout(0.2)(x31)

    x22 = conv2dtranspose(int(64 * 2))(x31)
    x22 = concatenate([x22, x20, x21])
    x22 = conv2d(filters = int(64 * 2))(x22)
    x22 = BatchNormalization()(x22)
    x22 = LeakyReLU(0.01)(x22)
    x22 = conv2d(filters = int(64 * 2))(x22)
    x22 = BatchNormalization()(x22)
    x22 = LeakyReLU(0.01)(x22)
    x22 = Dropout(0.2)(x22)

    x13 = conv2dtranspose(int(32 * 2))(x22)
    x13 = concatenate([x13, x10, x11, x12])
    x13 = conv2d(filters = int(32 * 2))(x13)
    x13 = BatchNormalization()(x13)
    x13 = LeakyReLU(0.01)(x13)
    x13 = conv2d(filters = int(32 * 2))(x13)
    x13 = BatchNormalization()(x13)
    x13 = LeakyReLU(0.01)(x13)
    x13 = Dropout(0.2)(x13)

    x04 = conv2dtranspose(int(16 * 2))(x13)
    x04 = concatenate([x04, x00, x01, x02, x03], axis=3)
    x04 = conv2d(filters = int(16 * 2))(x04)
    x04 = BatchNormalization()(x04)
    x04 = LeakyReLU(0.01)(x04)
    x04 = conv2d(filters = int(16 * 2))(x04)
    x04 = BatchNormalization()(x04)
    x04 = LeakyReLU(0.01)(x04)
    x04 = Dropout(0.2)(x04)

    output = Conv2D(config['num_classes'], kernel_size = (1, 1), activation = 'softmax')(x04)
 
    model = Model(inputs=[input], outputs=[output])
    
    return model


# Keras unet collection
def kuc_vnet(config):
    
    """
        Summary:
            Create VNET from keras unet collection library model object
        Arguments: 
            Model configuration from config.yaml
        Return:
            Keras.model object
    """

    model = kuc.vnet_2d((config['height'], config['width'], config['in_channels']), filter_num=[16, 32, 64, 128, 256], 
                        n_labels=config['num_classes'] ,res_num_ini=1, res_num_max=3, 
                        activation='PReLU', output_activation='Softmax', 
                        batch_norm=True, pool=False, unpool=False, name='vnet')
    x = model.layers[-2].output # fetch the last layer previous layer output
    
    output = Conv2D(config['num_classes'], kernel_size = (1,1), name="out", activation = 'softmax',dtype="float32")(x) # create new last layer
    model = Model(inputs = model.input, outputs=output)
    return model

def kuc_unet3pp(config):
    
    """
        Summary:
            Create UNET 3++ from keras unet collection library model object
        Arguments: 
            Model configuration from config.yaml
        Return:
            Keras.model object
    """
    
    model = kuc.unet_3plus_2d((config['height'], config['width'], config['in_channels']), 
                                n_labels=config['num_classes'], filter_num_down=[64, 128, 256, 512], 
                                filter_num_skip='auto', filter_num_aggregate='auto', 
                                stack_num_down=2, stack_num_up=1, activation='ReLU', output_activation='Softmax',
                                batch_norm=True, pool='max', unpool=False, deep_supervision=True, name='unet3plus')
    x = model.layers[-2].output # fetch the last layer previous layer output
    
    output = Conv2D(config['num_classes'], kernel_size = (1,1), name="out", activation = 'softmax',dtype="float32")(x) # create new last layer
    model = Model(inputs = model.input, outputs=output)
    return model

def kuc_r2unet(config):
    
    """
        Summary:
            Create R2UNET from keras unet collection library model object
        Arguments: 
            Model configuration from config.yaml
        Return:
            Keras.model object
    """

    model = kuc.r2_unet_2d((config['height'], config['width'], config['in_channels']), [64, 128, 256, 512], 
                            n_labels=config['num_classes'],
                             stack_num_down=2, stack_num_up=1, recur_num=2,
                            activation='ReLU', output_activation='Softmax', 
                            batch_norm=True, pool='max', unpool='nearest', name='r2unet')
    x = model.layers[-2].output # fetch the last layer previous layer output
    
    output = Conv2D(config['num_classes'], kernel_size = (1,1), name="out", activation = 'softmax',dtype="float32")(x) # create new last layer
    model = Model(inputs = model.input, outputs=output)
    return model

def kuc_unetpp(config):
    
    """
        Summary:
            Create UNET++ from keras unet collection library model object
        Arguments: 
            Model configuration from config.yaml
        Return:
            Keras.model object
    """
    
    model = kuc.unet_plus_2d((config['height'], config['width'], config['in_channels']), [64, 128, 256, 512], 
                            n_labels=config['num_classes'],
                            stack_num_down=2, stack_num_up=1, recur_num=2,
                            activation='ReLU', output_activation='Softmax', 
                            batch_norm=True, pool='max', unpool='nearest', name='r2unet')
    x = model.layers[-2].output # fetch the last layer previous layer output
    
    output = Conv2D(config['num_classes'], kernel_size = (1,1), name="out", activation = 'softmax',dtype="float32")(x) # create new last layer
    model = Model(inputs = model.input, outputs=output)
    return model


def kuc_restunet(config):
    
    """
        Summary:
            Create RESTUNET from keras unet collection library model object
        Arguments: 
            Model configuration from config.yaml
        Return:
            Keras.model object
    """

    model = kuc.resunet_a_2d((config['height'], config['width'], config['in_channels']), [32, 64, 128, 256, 512, 1024], 
                            dilation_num=[1, 3, 15, 31], 
                            n_labels=config['num_classes'], aspp_num_down=256, aspp_num_up=128, 
                            activation='ReLU', output_activation='Softmax', 
                            batch_norm=True, pool=False, unpool='nearest', name='resunet')
    x = model.layers[-2].output # fetch the last layer previous layer output
    
    output = Conv2D(config['num_classes'], kernel_size = (1,1), name="out", activation = 'softmax',dtype="float32")(x) # create new last layer
    model = Model(inputs = model.input, outputs=output)
    return model


def kuc_transunet(config):
    
    """
        Summary:
            Create TENSNET from keras unet collection library model object
        Arguments: 
            Model configuration from config.yaml
        Return:
            Keras.model object
    """
    
    model = kuc.transunet_2d((config['height'], config['width'], config['in_channels']), filter_num=[64, 128, 256, 512],
                            n_labels=config['num_classes'], stack_num_down=2, stack_num_up=2,
                            embed_dim=768, num_mlp=3072, num_heads=12, num_transformer=12,
                            activation='ReLU', mlp_activation='GELU', output_activation='Softmax', 
                            batch_norm=True, pool=True, unpool='bilinear', name='transunet')
    x = model.layers[-2].output # fetch the last layer previous layer output
    
    output = Conv2D(config['num_classes'], kernel_size = (1,1), name="out", activation = 'softmax',dtype="float32")(x) # create new last layer
    model = Model(inputs = model.input, outputs=output)
    return model


def kuc_swinnet(config):
    
    """
        Summary:
            Create SWINNET from keras unet collection library model object
        Arguments: 
            Model configuration from config.yaml
        Return:
            Keras.model object
    """
    
    model = kuc.swin_unet_2d((config['height'], config['width'], config['in_channels']), filter_num_begin=64, 
                            n_labels=config['num_classes'], depth=4, stack_num_down=2, stack_num_up=2, 
                            patch_size=(2, 2), num_heads=[4, 8, 8, 8], window_size=[4, 2, 2, 2], num_mlp=512, 
                            output_activation='Softmax', shift_window=True, name='swin_unet')
    x = model.layers[-2].output # fetch the last layer previous layer output
    
    output = Conv2D(config['num_classes'], kernel_size = (1,1), name="out", activation = 'softmax',dtype="float32")(x) # create new last layer
    model = Model(inputs = model.input, outputs=output)
    return model


def kuc_u2net(config):
    
    """
        Summary:
            Create U2NET from keras unet collection library model object
        Arguments: 
            Model configuration from config.yaml
        Return:
            Keras.model object
    """
    
    model = kuc.u2net_2d((config['height'], config['width'], config['in_channels']), n_labels=config['num_classes'], 
                            filter_num_down=[64, 128, 256, 512], filter_num_up=[64, 64, 128, 256], 
                            filter_mid_num_down=[32, 32, 64, 128], filter_mid_num_up=[16, 32, 64, 128], 
                            filter_4f_num=[512, 512], filter_4f_mid_num=[256, 256], 
                            activation='ReLU', output_activation='Softmax', 
                            batch_norm=True, pool=False, unpool=False, deep_supervision=True, name='u2net')
    x = model.layers[-2].output # fetch the last layer previous layer output
    
    output = Conv2D(config['num_classes'], kernel_size = (1,1), name="out", activation = 'softmax',dtype="float32")(x) # create new last layer
    model = Model(inputs = model.input, outputs=output)
    return model 



def kuc_attunet(config):
    
    """
        Summary:
            Create ATTENTION UNET from keras unet collection library model object
        Arguments: 
            Model configuration from config.yaml
        Return:
            Keras.model object
    """
  
    model = kuc.att_unet_2d((config['height'], config['width'], config['in_channels']), [64, 128, 256, 512], 
                            n_labels=config['num_classes'],
                            stack_num_down=2, stack_num_up=2,
                            activation='ReLU', atten_activation='ReLU', attention='add', output_activation=None, 
                            batch_norm=True, pool=False, unpool='bilinear', name='attunet')
    x = model.layers[-2].output # fetch the last layer previous layer output
    
    output = Conv2D(config['num_classes'], kernel_size = (1,1), name="out", activation = 'softmax',dtype="float32")(x) # create new last layer
    model = Model(inputs = model.input, outputs=output)
    return model


# Segmentation Models unet/linknet/fpn/pspnet
def sm_unet(config):
    
    """
        Summary:
            Create UNET from segmentation models library model object
        Arguments: 
            Model configuration from config.yaml
        Return:
            Keras.model object
    """

    model = sm.Unet(backbone_name='efficientnetb0', input_shape=(config['height'], config['width'], config['in_channels']),
                    classes = config['num_classes'], activation='softmax',
                    encoder_weights=None, weights=None)
    x = model.layers[-2].output # fetch the last layer previous layer output
    
    output = Conv2D(config['num_classes'], kernel_size = (1,1), name="out", activation = 'softmax',dtype="float32")(x) # create new last layer
    model = Model(inputs = model.input, outputs=output)
    return model


def sm_linknet(config):
    """
        Summary:
            Create LINKNET from segmentation models library model object
        Arguments: 
            Model configuration from config.yaml
        Return:
            Keras.model object
    """

    model = sm.Linknet(backbone_name='efficientnetb0', input_shape=(config['height'], config['width'], config['in_channels']),
                    classes = config['num_classes'], activation='softmax',
                    encoder_weights=None, weights=None)
    x = model.layers[-2].output # fetch the last layer previous layer output
    
    output = Conv2D(config['num_classes'], kernel_size = (1,1), name="out", activation = 'softmax',dtype="float32")(x) # create new last layer
    model = Model(inputs = model.input, outputs=output)
    return model


def sm_fpn(config):
    """
        Summary:
            Create FPN from segmentation models library model object
        Arguments: 
            Model configuration from config.yaml
        Return:
            Keras.model object
    """

    model = sm.FPN(backbone_name='efficientnetb0', input_shape=(config['height'], config['width'], config['in_channels']),
                    classes = config['num_classes'], activation='softmax',
                    encoder_weights=None, weights=None)
    x = model.layers[-2].output # fetch the last layer previous layer output
    
    output = Conv2D(config['num_classes'], kernel_size = (1,1), name="out", activation = 'softmax',dtype="float32")(x) # create new last layer
    model = Model(inputs = model.input, outputs=output)
    return model


def sm_pspnet(config):
    """
        Summary:
            Create PSPNET from segmentation models library model object
        Arguments: 
            Model configuration from config.yaml
        Return:
            Keras.model object
    """

    model = sm.PSPNet(backbone_name='efficientnetb0', input_shape=(config['height'], config['width'], config['in_channels']),
                    classes = config['num_classes'], activation='softmax', downsample_factor=8,
                    encoder_weights=None, weights=None)
    x = model.layers[-2].output # fetch the last layer previous layer output
    
    output = Conv2D(config['num_classes'], kernel_size = (1,1), name="out", activation = 'softmax',dtype="float32")(x) # create new last layer
    model = Model(inputs = model.input, outputs=output)
    return model



# Transfer Learning
# ----------------------------------------------------------------------------------------------

def get_model_transfer_lr(model, num_classes):
    """
    Summary:
        create new model object for transfer learning
    Arguments:
        model (object): keras.Model class object
        num_classes (int): number of class
    Return:
        model (object): keras.Model class object
    """


    x = model.layers[-2].output # fetch the last layer previous layer output
    
    output = Conv2D(num_classes, kernel_size = (1,1), name="out", activation = 'softmax')(x) # create new last layer
    model = Model(inputs = model.input, outputs=output) 
    
    # freeze all model layer except last layer
    for layer in model.layers[:-1]:
        layer.trainable = False
    
    return model

def ex_mnet(config):
    #Build the model
    inputs = Input((config['height'], config['width'], config['in_channels']))
    
    #Contraction path
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = Dropout(0.2)(c1)  # Original 0.1
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.2)(c2)  # Original 0.1
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
     
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
     
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.2)(c5)  # Original 0.1
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    p5 = MaxPooling2D((2, 2))(c5)
     
    c6 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p5)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
    p6 = MaxPooling2D((2, 2))(c6)

    c7 = Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p6)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
    p7 = MaxPooling2D((2, 2))(c7)

    c8 = Conv2D(2048, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p7)
    c8 = Dropout(0.2)(c8)
    c8 = Conv2D(2048, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
    p8 = MaxPooling2D((2, 2))(c8)
     
    c9 = Conv2D(4096, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p8)
    c9 = Dropout(0.3)(c9)
    c9 = Conv2D(4096, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
    

    # Expansive path

    u10 = Conv2DTranspose(2048, (2, 2), strides=(2, 2), padding='same')(c9)
    u10 = concatenate([u10, c8])
    c10 = Conv2D(2048, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u10)
    c10 = Dropout(0.2)(c10)
    c10 = Conv2D(2048, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c10)

    u11 = Conv2DTranspose(1024, (2, 2), strides=(2, 2), padding='same')(c10)
    u11 = concatenate([u11, c7])
    c11 = Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u11)
    c11 = Dropout(0.2)(c11)
    c11 = Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c11)

    u12 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c11)
    u12 = concatenate([u12, c6])
    c12 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u12)
    c12 = Dropout(0.2)(c12)
    c12 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c12)
    
    u13 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c12)
    u13 = concatenate([u13, c5])
    c13 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u13)
    c13 = Dropout(0.2)(c13)
    c13 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c13)
        
    u14 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c13)
    u14 = concatenate([u14, c4])
    c14 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u14)
    c14 = Dropout(0.2)(c14)
    c14 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c14)
     
    u15 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c14)
    u15 = concatenate([u15, c3])
    c15 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u15)
    c15 = Dropout(0.2)(c15)
    c15 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c15)
     
    u16 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c15)
    u16 = concatenate([u16, c2])
    c16 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u16)
    c16 = Dropout(0.2)(c16)  # Original 0.1
    c16 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c16)
     
    u17 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c16)
    u17 = concatenate([u17, c1], axis=3)
    c17 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u17)
    c17 = Dropout(0.2)(c17)  # Original 0.1
    c17 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c17)
     
    outputs = Conv2D(config['num_classes'], (1, 1), activation='softmax')(c17)
     
    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model

#PLANET model
def wnet(config):
    no_layer = 0
    inp_size = config["height"]
    start_filter = 16
    while inp_size>=8:
        no_layer += 1
        inp_size = inp_size / 2
    print(no_layer)
    encoder = {}
    inputs = Input((config['height'], config['width'], config['in_channels']))
    for i in range(no_layer):
        if i == 0:
            encoder["enc_{}_0".format(i)] = Conv2D(start_filter, (3, 3), name = "enc_{}_0".format(i),activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
        else:
            encoder["enc_{}_0".format(i)] = Conv2D(start_filter, (3, 3), name = "enc_{}_0".format(i), activation='relu', kernel_initializer='he_normal', padding='same')(encoder["mp_{}".format(i-1)])
        encoder["enc_{}_1".format(i)] = Conv2D(start_filter, (3, 3), name = "enc_{}_1".format(i), activation='relu', kernel_initializer='he_normal', padding='same')(Dropout(0.2)(encoder["enc_{}_0".format(i)]))
        encoder["mp_{}".format(i)] = MaxPooling2D((2,2), name = "mp_{}".format(i))(encoder["enc_{}_1".format(i)])
        start_filter *= 2
    
    mid_1 = Conv2D(start_filter, (3, 3), name = "mid_1", activation='relu', kernel_initializer='he_normal', padding='same')(encoder["mp_{}".format(no_layer-1)])
    mid_drop = Dropout(0.3)(mid_1)
    mid_2 = Conv2D(start_filter, (3, 3), name = "mid_2", activation='relu', kernel_initializer='he_normal', padding='same')(mid_drop)

    start_filter = start_filter / 2
    half_dec = {}
    for i in range(math.floor(no_layer/2)):
        print(i)
        if i == 0:
            half_dec["h_dec_T_{}".format(i)] = Conv2DTranspose(start_filter, (2, 2), name = "h_dec_T_{}".format(i), strides=(2, 2), padding='same')(mid_2)
        else:
            half_dec["h_dec_T_{}".format(i)] = Conv2DTranspose(start_filter, (2, 2), name = "h_dec_T_{}".format(i), strides=(2, 2), padding='same')(half_dec["h_dec_{}_1".format(i-1)])
        half_dec["h_cc_{}".format(i)] = concatenate([half_dec["h_dec_T_{}".format(i)], encoder["enc_{}_1".format(no_layer-i-1)]], axis=3)
        half_dec["h_dec_{}_0".format(i)] = Conv2D(start_filter, (3, 3), name = "h_dec_{}_0".format(i), activation='relu', kernel_initializer='he_normal', padding='same')(half_dec["h_cc_{}".format(i)])
        half_dec["h_dec_{}_1".format(i)] = Conv2D(start_filter, (3, 3), name = "h_dec_{}_1".format(i), activation='relu', kernel_initializer='he_normal', padding='same')(Dropout(0.2)(half_dec["h_dec_{}_0".format(i)]))
        start_filter = start_filter / 2
    half_dec["h_dec_T_{}".format(math.floor(no_layer/2))] = Conv2DTranspose(start_filter, (2, 2), name = "h_dec_T_{}".format(math.floor(no_layer/2)), strides=(2, 2), padding='same')(half_dec["h_dec_{}_1".format(math.floor(no_layer/2)-1)])


    p_mid_1 = Conv2D(start_filter, (3, 3), name = "p_mid_1", activation='relu', kernel_initializer='he_normal', padding='same')(half_dec["h_dec_T_{}".format(math.floor(no_layer/2))])
    p_mid_drop = Dropout(0.3)(p_mid_1)
    p_mid_2 = Conv2D(start_filter, (3, 3), name = "p_mid_2", activation='relu', kernel_initializer='he_normal', padding='same')(p_mid_drop)
    mid_pool = MaxPooling2D((2,2), name = "mid_mp")(p_mid_2)

    half_enc = {}
    start_filter *= 2
    for i in range(math.floor(no_layer/2)):
        if i == 0:
            half_enc["h_enc_{}_0".format(i)] = Conv2D(start_filter, (3, 3), name = "h_enc_{}_0".format(i),activation='relu', kernel_initializer='he_normal', padding='same')(mid_pool)
        else:
            half_enc["h_enc_{}_0".format(i)] = Conv2D(start_filter, (3, 3), name = "h_enc_{}_0".format(i), activation='relu', kernel_initializer='he_normal', padding='same')(half_enc["h_mp_{}".format(i-1)])
        half_enc["h_enc_{}_1".format(i)] = Conv2D(start_filter, (3, 3), name = "h_enc_{}_1".format(i), activation='relu', kernel_initializer='he_normal', padding='same')(Dropout(0.2)(half_enc["h_enc_{}_0".format(i)]))
        half_enc["h_mp_{}".format(i)] = MaxPooling2D((2,2), name = "h_mp_{}".format(i))(half_enc["h_enc_{}_1".format(i)])
        start_filter *= 2
    

    l_mid_1 = Conv2D(start_filter, (3, 3), name = "l_mid_1", activation='relu', kernel_initializer='he_normal', padding='same')(half_enc["h_mp_{}".format(math.floor(no_layer/2)-1)])
    l_mid_drop = Dropout(0.3)(l_mid_1)
    l_mid_2 = Conv2D(start_filter, (3, 3), name = "l_mid_2", activation='relu', kernel_initializer='he_normal', padding='same')(l_mid_drop)


    
    decoder = {}
    for i in range(no_layer):
        if i == 0:
            decoder["dec_T_{}".format(i)] = Conv2DTranspose(start_filter, (2, 2), name = "dec_T_{}".format(i), strides=(2, 2), padding='same')(l_mid_2)
        else:
            decoder["dec_T_{}".format(i)] = Conv2DTranspose(start_filter, (2, 2), name = "dec_T_{}".format(i), strides=(2, 2), padding='same')(decoder["dec_{}_1".format(i-1)])
        
        if i < math.floor(no_layer/2):
            decoder["cc_{}".format(i)] = concatenate([decoder["dec_T_{}".format(i)], half_enc["h_enc_{}_1".format(math.floor(no_layer/2)-i-1)]], axis=3)
        else:
            decoder["cc_{}".format(i)] = concatenate([decoder["dec_T_{}".format(i)], encoder["enc_{}_1".format(no_layer-i-1)]], axis=3)
        decoder["dec_{}_0".format(i)] = Conv2D(start_filter, (3, 3), name = "dec_{}_0".format(i), activation='relu', kernel_initializer='he_normal', padding='same')(decoder["cc_{}".format(i)])
        decoder["dec_{}_1".format(i)] = Conv2D(start_filter, (3, 3), name = "dec_{}_1".format(i), activation='relu', kernel_initializer='he_normal', padding='same')(Dropout(0.2)(decoder["dec_{}_0".format(i)]))
        start_filter = start_filter / 2
    
    outputs = Conv2D(config['num_classes'], (1, 1), activation='softmax', dtype='float32')(decoder["dec_{}_1".format(no_layer-1)])
    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model




image_size = 512  # We'll resize input images to this size
patch_size = 16  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2
projection_dim = 192
num_heads = 4
input_shape = (512, 512, 3)
num_class = 2
transformer_units = [
    projection_dim * 3,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 8


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
    def get_config(self):
        return {"patch_size": self.patch_size}


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded
    
    def get_config(self):
        return {"num_patches": self.num_patches,
                "projection_dim": self.projection_dim}


class DecoderLinear(layers.Layer):
    def __init__(self, n_cls, patch_size):
        super().__init__()

        self.patch_size = patch_size
        self.n_cls = n_cls
        self.im_size = (512, 512)

        self.head = layers.Dense(n_cls)

    def call(self, x):
        H, W = self.im_size
        GS = H // self.patch_size
        x = self.head(x)
        x = rearrange(x, "b (h w) c -> b h w c", h=GS)

        return x


class Block(layers.Layer):
    def __init__(self, num_heads, projection_dim,  transformer_units, dropout=0.1):
        super().__init__()

        self.dropout = dropout
        self.num_heads = num_heads
        self.projection_dim = projection_dim
        self.transformer_units = transformer_units

        self.x1 = layers.LayerNormalization(epsilon=1e-6)
        self.attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=dropout
        )
        self.x2 = layers.Add()
        self.x3 = layers.LayerNormalization(epsilon=1e-6)
        self.dense1 = layers.Dense(transformer_units, activation=tf.nn.gelu)
        self.drop = layers.Dropout(dropout)
        self.dense2 = layers.Dense(projection_dim, activation=tf.nn.gelu)
        self.x4 = layers.Add()

    def call(self, encoded_patches):
        # Layer normalization 1.
        tmp1 = self.x1(encoded_patches)

        # Create a multi-head attention layer.
        att_out = self.attention_output(tmp1, tmp1)

        # Skip connection 1.
        tmp2 = self.x2([att_out, encoded_patches])

        # Layer normalization 2.
        tmp3 = self.x3(tmp2)

        # MLP
        tmp3 = self.drop(self.dense1(tmp3))
        tmp3 = self.drop(self.dense2(tmp3))

        # Skip connection 2.
        encoded_patches = self.x4([tmp3,tmp2])
        return encoded_patches
    
    def get_config(self):
        return {"num_heads": self.num_heads,
                "projection_dim": self.projection_dim,
                "transformer_units": self.transformer_units,
                "dropout": self.dropout}


class MaskTransformer(layers.Layer):
    def __init__(
        self,
        n_cls,
        patch_size,
        d_encoder,
        n_layers,
        n_heads,
        d_model,
        dropout,
    ):
        super(MaskTransformer, self).__init__()
        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.n_cls = n_cls
        self.n_heads = n_heads
        self.dropout = dropout
        self.d_model = d_model
        self.scale = d_model ** -0.5
        self.im_size = (512, 512)

        self.blocks = [Block(n_heads, d_model, d_model*4, dropout) for _ in range(n_layers)]

        self.cls_emb = tf.Variable(tf.random.truncated_normal((1, n_cls, d_model), stddev=0.2, seed=123), trainable=False, name="cls_emb")
        self.proj_dec = layers.Dense(d_model)

        self.proj_patch = tf.Variable(self.scale * tf.random.normal((d_model, d_model)), name="proj_patch")
        self.proj_classes = tf.Variable(self.scale * tf.random.normal((d_model, d_model)), name="proj_classes")

        self.decoder_norm = layers.LayerNormalization(epsilon=1e-6)
        self.mask_norm = layers.LayerNormalization(epsilon=1e-6)

        #trunc_normal_(self.cls_emb, std=0.02)

    def call(self, x):
        H, W = self.im_size
        GS = H // self.patch_size

        x = self.proj_dec(x)
        cls_emb = tf.tile(self.cls_emb, [x.shape[0], 1, 1])
        x = tf.concat([x, cls_emb], 1) # x-> [None, 1024, 192] cls_emb-> [None, 2, 192]
        for blk in self.blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        patches, cls_seg_feat = x[:, : -self.n_cls], x[:, -self.n_cls :]
        patches = patches @ self.proj_patch
        cls_seg_feat = cls_seg_feat @ self.proj_classes

        patches = patches / tf.norm(patches, axis=-1, keepdims=True)
        cls_seg_feat = cls_seg_feat / tf.norm(cls_seg_feat, axis=-1, keepdims=True)

        masks = patches @ tf.transpose(cls_seg_feat, perm=[0,2,1])
        masks = self.mask_norm(masks)
        masks = rearrange(masks, "b (h w) n -> b h w n", h=int(GS))

        return masks

    def get_attention_map(self, x, layer_id):
        if layer_id >= self.n_layers or layer_id < 0:
            raise ValueError(
                f"Provided layer_id: {layer_id} is not valid. 0 <= {layer_id} < {self.n_layers}."
            )
        x = self.proj_dec(x)
        cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
        x = tf.concat([x, cls_emb], 1)
        for i, blk in enumerate(self.blocks):
            if i < layer_id:
                x = blk(x)
            else:
                return blk(x, return_attention=True)
    def get_config(self):
        return {"d_encoder": self.d_encoder,
                "patch_size": self.patch_size,
                "n_layers": self.n_layers,
                "n_cls": self.n_cls,
                "n_heads": self.n_heads,
                "dropout": self.dropout,
                "d_model": self.d_model,
                "scale": self.scale,
                "im_size": self.im_size}


def create_vit_classifier(config):
    inputs = layers.Input(shape=input_shape, batch_size=config["batch_size"])
    # Augment data.
    # Create patches.
    patches = Patches(patch_size)(inputs)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        encoded_patches = Block(num_heads, projection_dim, projection_dim*3, 0.1)(encoded_patches)

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    head = layers.Dense(1000)(representation)
    #logits = DecoderLinear(n_cls=num_class, patch_size=patch_size)(head)
    masks = MaskTransformer(n_cls=num_class, patch_size=patch_size, d_encoder=projection_dim, 
                            n_layers=2, n_heads=3, d_model=projection_dim, dropout=0.1)(head)
    logits = layers.UpSampling2D((16,16), interpolation="bilinear")(masks)
    

    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model



# Get model
# ----------------------------------------------------------------------------------------------

def get_model(config):
    """
    Summary:
        create new model object for training
    Arguments:
        config (dict): Configuration directory
    Return:
        model (object): keras.Model class object
    """


    models = {'unet': unet,
              'fapnet': mod_unet,
              'ex_mnet':ex_mnet,
              'dncnn': DnCNN,
              'u2net': u2net,
              'vnet': vnet,
              'unet++': unet_plus_plus,
              'sm_unet':sm_unet,
              'sm_linknet':sm_linknet,
              'sm_fpn':sm_fpn,
              'sm_pspnet':sm_pspnet,
              'kuc_vnet':kuc_vnet,
              'kuc_unet3pp':kuc_unet3pp,
              'kuc_r2unet':kuc_r2unet,
              'kuc_unetpp':kuc_unetpp,
              'kuc_restunet':kuc_restunet,
              'kuc_tensnet':kuc_transunet,
              'kuc_swinnet':kuc_swinnet,
              'kuc_u2net':kuc_u2net,
              'kuc_attunet':kuc_attunet,
              'ad_unet':ad_unet,
              "transformer":create_vit_classifier
              }
    return models[config['model_name']](config)    

if __name__ == '__main__':
    
    model = vnet()
    model.summary()
