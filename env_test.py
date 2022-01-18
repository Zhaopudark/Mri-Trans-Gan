import tensorflow as tf
import datetime
import os
from tensorflow.keras import mixed_precision
# tf.config.experimental.enable_tensor_float_32_execution(False)
physical_devices = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.keras.utils.set_random_seed(1314)
# policy = mixed_precision.Policy('float64')
# mixed_precision.set_global_policy(policy)

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
# for i,item in enumerate(x_train):
#     print(i,item.shape,tf.reduce_mean(item))
#     if i == 10:
#         break
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    # tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])
# tf.random.set_seed(1000)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.1)
checkpoint_directory = "./tmp/training_checkpoints"
checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
checkpoint = tf.train.Checkpoint(model=model,optimizer=optimizer)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))
checkpoint.save(file_prefix=checkpoint_prefix)

# model.save_weights("./tmp/")
# model.load_weights("./tmp/")
# checkpoint = tf.train.Checkpoint(model)
# save_path = checkpoint.save('./tmp/training_checkpoints')
# print(save_path)
# checkpoint.restore("./tmp/training_checkpoints-1")
# tf.random.set_seed(1000)
# optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.0, beta_2=0.999, epsilon=1e-7)

model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# tf.keras.models.save_model(model=model,filepath="./tmp/training_checkpoints",overwrite=True, include_optimizer=True)
# loaded_model=tf.keras.models.load_model("./tmp/training_checkpoints")


# model.load_weights("./tmp/")
# model.save_weights("./tmp/")
# log_dir="testlogs/profile/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch = 3)
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = log_dir,
                                                    #   profile_batch = '500,510')

# model.fit(x_train, y_train, epochs=5,callbacks=[tensorboard_callback])
# model.evaluate(x_test,  y_test, verbose=2,callbacks=[tensorboard_callback])
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test,  y_test, verbose=2)

