---
title: tensflow2.0系列-神经网络模型拼接问题
published: 2026-08-22T17:56:00.000+08:00
updated: 2026-08-22T17:56:00.000+08:00
tags:
  - TensFlow
category: TensFlow2.x
draft: false
---
使用如下方法拼接即可：

```
def CNN(image_shape, soil_pm_shape, is_in=0):
    img_input = tf.keras.Input(shape=image_shape)
    soil_pm_input = tf.keras.Input(shape=soil_pm_shape)

    x = tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=2)(img_input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Flatten()(x)

    new_x = tf.keras.layers.concatenate([x, soil_pm_input], axis=1)

    x_1 = tf.keras.layers.Dense(256)(new_x) if is_in!=1 else tf.keras.layers.Dense(64)(new_x)
    x_1 = tf.keras.layers.BatchNormalization()(x_1)
    x_1 = tf.keras.layers.Activation('relu')(x_1)

    x_1 = tf.keras.layers.Dense(186)(x_1) if is_in!=1 else tf.keras.layers.Dense(2)(x_1)
    x_1 = tf.keras.layers.BatchNormalization()(x_1)
    out_put = tf.keras.layers.Activation('sigmoid')(x_1)

    model = tf.keras.Model(inputs=[img_input, soil_pm_input], outputs=out_put)
    return model
```

编译与训练如下所示：

```
model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='mean_squared_error'
    )
    model.fit(
        x=train_pm,
        y=train_bg,
        validation_data=[confirm_pm, confirm_bg],
        batch_size=batch_size,
        epochs=epochs,
        verbose=1
    )
```
