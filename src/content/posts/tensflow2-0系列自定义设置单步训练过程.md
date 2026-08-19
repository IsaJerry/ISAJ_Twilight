---
title: tensflow2.0系列自定义设置单步训练过程
published: 2026-08-19T17:56:00.000+08:00
updated: 2026-08-19T18:04:00.000+08:00
tags:
  - TensFlow
category: TensFlow2.x
draft: false
---
自定义训练训练步骤可以单独设置一个tensflow函数，示例如下：

```
@tf.function
def train_step(x, inverse_net, p_model, bg_maxs, bg_mins, soil_maxs, soil_mins,
               filling_rate_maxs, filling_rate_mins, input_pm):
    with tf.GradientTape() as tape:
        filling_rate, constant = inverse_net(x, training=True)
        n_filling = (filling_rate-filling_rate_mins) / (filling_rate_maxs-filling_rate_mins)
        n_soil = (input_pm-soil_mins) / (soil_maxs-soil_mins)
        p_input = tf.concat([n_soil, n_filling], axis=1)
        p_bg = p_model(p_input, training=False)
        p_bg = p_bg*(bg_maxs-bg_mins) + bg_mins
        p_bg = p_bg/constant
        loss = tf.reduce_mean(tf.square(p_bg[:, :2]-x))
    grads = tape.gradient(loss, inverse_net.trainable_variables)
    return loss, grads
```

以下代码放在tf函数外：

```
optimizer.apply_gradients(zip(grads, inverse_net.trainable_variables))
```

具体调用方法：

```
loss, grads = train_step(input_bg, inverse_net, model, bg_maxs, bg_mins, soil_maxs, soil_mins,
                              filling_rate_maxs, filling_rate_mins, input_pm)
            optimizer.apply_gradients(zip(grads, inverse_net.trainable_variables))
```
