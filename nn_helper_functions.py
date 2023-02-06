import tensorflow as tf

def relu_bn(inputs: tf.Tensor) -> tf.Tensor:
    relu = tf.keras.layers.ReLU()(inputs)
    bn = tf.keras.layers.BatchNormalization()(relu)
    return bn

def residual_block(x: tf.Tensor, downsample: bool, filters: int, kernel_size: int = 3, squeeze_and_excitation=False, se_ration=2) -> tf.Tensor:
    y = tf.keras.layers.Conv2D(kernel_size=kernel_size,
               strides= (1 if not downsample else 2),
               filters=filters,
               padding="same")(x)
    y = relu_bn(y)
    y = tf.keras.layers.Conv2D(kernel_size=kernel_size,
               strides=1,
               filters=filters,
               padding="same")(y)

    if downsample:
        x = tf.keras.layers.Conv2D(kernel_size=1,
                   strides=2,
                   filters=filters,
                   padding="same")(x)

    # se layer
    # https://towardsdatascience.com/squeeze-and-excitation-networks-9ef5e71eacd7
    if squeeze_and_excitation:
        x_se = tf.keras.layers.GlobalAveragePooling2D()(x)
        x_se = tf.keras.layers.Dense(filters // se_ration, activation='relu')(x_se)
        x_se = tf.keras.layers.Dense(filters, activation='sigmoid')(x_se)
        # t =  tf.keras.layers.multiply()([t, x_se])

        # fix shape (needed for broadcasting multiple samples)
        x_se = tf.expand_dims(tf.expand_dims(x_se, axis=1), axis=1)
        x = x * x_se

    # res tower skip connection
    out = tf.keras.layers.Add()([x, y])
    out = relu_bn(out)

    return out