---
title: tensflow2.0系列-自动训练系列
published: 2026-08-22T17:42:00.000+08:00
updated: 2026-08-22T17:42:00.000+08:00
tags:
  - TensFlow
category: TensFlow2.x
draft: false
---
**第一步：**首先构建神经网络模型，有如下方法：

1.`tf.keras.Sequential([ ])`方法：

```
def MLP(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(512, input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),

        tf.keras.layers.Dense(256),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),

        tf.keras.layers.Dense(128),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),

        tf.keras.layers.Dense(2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('sigmoid')
    ])
    return model
```

2. `tf.keras.Model()`方法：

```
def MLP(input_dimension, output_bgs, model_name='Pretrain_MLP_P'):
    inputs = tf.keras.Input(shape=(input_dimension,))
    x = tf.keras.layers.Dense(1024)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Dense(512)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Dense(256)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Dense(256)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Dense(128)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Dense(64)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Dense(output_bgs)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    outputs = tf.keras.layers.Activation("sigmoid")(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name=model_name
    )
    return model
```

**第二步：**实例化神经网络模型并编译：

```
model = MLP((5,))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='mean_squared_error'
    )
```

**第三步：**训练：

`x`与`y`分别是输入输出数据，`validation_data`是验证数据该选项可选，`epochs`是训练次数又叫迭代次数，`batch_size`是在`epochs`次数中可以分成`n`组，每组`batch_size`个，`verbose`是进度条，`1`是显示。

```
history = model.fit(
        x=train_pm,
        y=train_bg,
        validation_data=(confirm_pm, confirm_bg),
        epochs=epochs,
        batch_size=epochs,
        verbose=1
    )
```
