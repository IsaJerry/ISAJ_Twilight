---
title: tensflow2.0系列-反向训练参数问题
published: 2026-08-22T17:13:00.000+08:00
updated: 2026-08-22T17:13:00.000+08:00
tags:
  - TensFlow
category: TensFlow2.x
draft: false
---
在2.0中进行TNN训练反向设置参数时，由于是自定义训练步骤，且每步输入的`batch_size = 1` 的数据时，在参数设计神经网络中不出现`tf.keras.layers.BatchNormalization()`时训练效果会更好，如下图所示

```
def inverse_network(input_dimension, model_name, filling_up_limit, filling_low_limit, p_c_up_limit, p_c_low_limit):
    inputs = tf.keras.Input(shape=(input_dimension, ))

    x = tf.keras.layers.Dense(16)(inputs)
    # x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('tanh')(x)

    # f_output = tf.keras.layers.Dense(1)(x)
    # filling = tf.keras.layers.BatchNormalization()(f_output)
    filling = tf.keras.layers.Dense(1)(x)
    filling_concrete = ((tf.math.sin(filling)+1)/2*
                        (filling_up_limit - filling_low_limit) + filling_low_limit)
    filling_soil = 1 - filling_concrete
    filling_rate = tf.concat([filling_concrete, filling_soil], axis=1)

    # p_c_output = tf.keras.layers.Dense(1)(x)
    # p_c = tf.keras.layers.BatchNormalization()(p_c_output)
    p_c = tf.keras.layers.Dense(1)(x)
    constant = (tf.math.sin(p_c)+1)/2*(p_c_up_limit - p_c_low_limit) + p_c_low_limit

    return tf.keras.Model(
        inputs=inputs,
        outputs=[filling_rate, constant],
        name=model_name
    )
```

并且在训练时应直接在步骤中输出神经网络的输出，不应再此调用`inverse_net.predict(input_bg)`否者相当于再次训练了一轮。如下所示：：

```
    for i in range(bandgaps.shape[0]):
        optimizer = tf.keras.optimizers.Adam(1e-3)
        input_bg = bandgaps[i:i+1, :]
        input_pm = parameters[i:i+1, :]
        flag = False
        for epoch in range(epochs):
            loss, grads,  filling_rate, constant = train_step(input_bg, inverse_net, model, bg_maxs, bg_mins, soil_maxs, soil_mins,
                              filling_rate_maxs, filling_rate_mins, input_pm)
            optimizer.apply_gradients(zip(grads, inverse_net.trainable_variables))
            loss = float(loss.numpy())
            if epoch > 100 and loss < 1e-2:
                flag =True
                print(f"Sample {i:02d} | Epochs: {epoch + 1:04d} | Loss: {np.round(loss, 6)}")
                break

        # filling_rate, constant = inverse_net.predict(input_bg)
        d_filling_rate[i] = filling_rate
        d_p_c[i] = constant
        d_soil[i] = input_pm
        d_bg[i] = input_bg
        d_loss[i] = loss
```

这样设置后效果会提升很多：

带`tf.keras.layers.BatchNormalization()`的效果：

```
1/1 [==============================] - 0s 83ms/step
Sample 00 | Epochs: 2000 | Loss: 5.281638e+02
1/1 [==============================] - 0s 64ms/step
Sample 01 | Epochs: 2000 | Loss: 3.181857e+02
1/1 [==============================] - 0s 77ms/step
Sample 02 | Epochs: 2000 | Loss: 4.975657e-01
1/1 [==============================] - 0s 65ms/step
Sample 03 | Epochs: 2000 | Loss: 4.303170e+02
Sample 04 | Epochs: 1524 | Loss: 0.009939
1/1 [==============================] - 0s 67ms/step
WARNING:tensorflow:5 out of the last 5 calls to <function Model.make_predict_function.<locals>.predict_function at 0x00000205D7B2B0D0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
1/1 [==============================] - 0s 75ms/step
Sample 05 | Epochs: 2000 | Loss: 5.356166e+02
WARNING:tensorflow:6 out of the last 6 calls to <function Model.make_predict_function.<locals>.predict_function at 0x00000205D7E7EC10> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
1/1 [==============================] - 0s 65ms/step
Sample 06 | Epochs: 2000 | Loss: 2.695414e+02
Sample 07 | Epochs: 1119 | Loss: 0.009881
```

不带`tf.keras.layers.BatchNormalization()`的效果：

```
Sample 00 | Epochs: 2000 | Loss: 5.281638e+02
Sample 01 | Epochs: 2000 | Loss: 3.181929e+02
Sample 02 | Epochs: 2000 | Loss: 3.027072e-01
Sample 03 | Epochs: 2000 | Loss: 7.938098e-01
Sample 04 | Epochs: 0303 | Loss: 0.009693
Sample 05 | Epochs: 0263 | Loss: 0.00985
Sample 06 | Epochs: 0256 | Loss: 0.009734
Sample 07 | Epochs: 0210 | Loss: 0.009648
Sample 08 | Epochs: 0126 | Loss: 0.006857
Sample 09 | Epochs: 0129 | Loss: 0.007794
Sample 10 | Epochs: 0334 | Loss: 0.009692
Sample 11 | Epochs: 0102 | Loss: 0.002452
Sample 12 | Epochs: 0274 | Loss: 0.009767
```
