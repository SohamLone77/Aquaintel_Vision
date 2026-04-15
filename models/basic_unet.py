"""
Basic U-Net model for image enhancement.
"""

from tensorflow.keras import Model, layers


def conv_block(input_tensor, num_filters, kernel_size=3):
    """Convolutional block: Conv -> BatchNorm -> ReLU."""
    x = layers.Conv2D(num_filters, kernel_size, padding="same")(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x


def encoder_block(input_tensor, num_filters):
    """Encoder block: Two conv layers + MaxPool."""
    x = conv_block(input_tensor, num_filters)
    x = conv_block(x, num_filters)
    skip = x
    x = layers.MaxPool2D(pool_size=(2, 2))(x)
    return x, skip


def decoder_block(input_tensor, skip_tensor, num_filters):
    """Decoder block: UpConv -> Concatenate -> Two conv layers."""
    x = layers.Conv2DTranspose(num_filters, kernel_size=2, strides=2, padding="same")(input_tensor)
    x = layers.Concatenate()([x, skip_tensor])
    x = conv_block(x, num_filters)
    x = conv_block(x, num_filters)
    return x


def build_basic_unet(input_shape=(256, 256, 3)):
    """Build basic U-Net model."""
    inputs = layers.Input(shape=input_shape)

    # Encoder
    x1, skip1 = encoder_block(inputs, 32)
    x2, skip2 = encoder_block(x1, 64)
    x3, skip3 = encoder_block(x2, 128)
    x4, skip4 = encoder_block(x3, 256)

    # Bottleneck
    bottleneck = conv_block(x4, 512)
    bottleneck = conv_block(bottleneck, 512)

    # Decoder
    d4 = decoder_block(bottleneck, skip4, 256)
    d3 = decoder_block(d4, skip3, 128)
    d2 = decoder_block(d3, skip2, 64)
    d1 = decoder_block(d2, skip1, 32)

    # Output
    outputs = layers.Conv2D(3, 1, activation="sigmoid")(d1)

    model = Model(inputs, outputs, name="Basic_U-Net")
    return model