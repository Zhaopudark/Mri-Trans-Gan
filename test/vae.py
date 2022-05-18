import tensorflow as tf 
import numpy as np 
from matplotlib import pyplot as plt
from tensorflow.keras import layers

physical_devices = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.keras.utils.set_random_seed(0)
tf.config.experimental.enable_op_determinism()

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Encoder(layers.Layer):
    """Maps MNIST digits to a triplet (z_mean, z_log_var, z)."""

    def __init__(self, latent_dim=32, intermediate_dim=64, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.dense_proj = layers.Dense(intermediate_dim, activation=tf.nn.relu)
        self.dense_mean = layers.Dense(latent_dim)
        self.dense_log_var = layers.Dense(latent_dim)
        self.sampling = Sampling()

    def call(self, inputs):
        x = self.dense_proj(inputs)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z

class Decoder(layers.Layer):
    """Converts z, the encoded digit vector, back into a readable digit."""

    def __init__(self, original_dim, intermediate_dim=64, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.dense_proj = layers.Dense(intermediate_dim, activation=tf.nn.relu)
        self.dense_output = layers.Dense(original_dim, activation=tf.nn.sigmoid)

    def call(self, inputs):
        x = self.dense_proj(inputs)
        return self.dense_output(x)

class VariationalAutoEncoder(layers.Layer):
    """Combines the encoder and decoder into an end-to-end model for training."""

    def __init__(self, original_dim, intermediate_dim=64, latent_dim=32, **kwargs):
        super(VariationalAutoEncoder, self).__init__(**kwargs)
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim=latent_dim, intermediate_dim=intermediate_dim)
        self.decoder = Decoder(original_dim, intermediate_dim=intermediate_dim)

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        # Add KL divergence regularization loss.
        kl_loss = -0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
        )
        self.add_loss(kl_loss)
        return reconstructed

vae = VariationalAutoEncoder(original_dim=784, intermediate_dim=64, latent_dim=32)

# Loss and optimizer.
loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

# Prepare a dataset.
(x_train, _), _ = tf.keras.datasets.mnist.load_data()
dataset = tf.data.Dataset.from_tensor_slices(
    x_train.reshape(60000, 784).astype("float32") / 255
)
dataset = dataset.shuffle(buffer_size=1024).batch(32)


@tf.function
def training_step(x):
    with tf.GradientTape() as tape:
        reconstructed = vae(x)  # Compute input reconstruction.
        # Compute loss.
        loss = loss_fn(x, reconstructed)
        loss += sum(vae.losses)  # Add KLD term.
    optimizer.minimize(loss,vae.trainable_weights,tape=tape)
    return loss


losses = []  # Keep track of the losses over time.
for step, x in enumerate(dataset):
    loss = training_step(x)
    # Logging.
    losses.append(float(loss))
    if step % 100 == 0:
        print("Step:", step, "Loss:", sum(losses) / len(losses))

    # Stop after 1000 steps.
    # Training the model to convergence is left
    # as an exercise to the reader.
    if step >= 1000:
        break

import tensorflow_probability as tfp
def plot_latent_images(model, n, digit_size=28):
    """Plots n x n digit images decoded from the latent space."""

    norm = tfp.distributions.Normal(0, 1)
    grid_x = norm.quantile(np.linspace(0.05, 0.95, n))
    grid_y = norm.quantile(np.linspace(0.05, 0.95, n))
    image_width = digit_size*n
    image_height = image_width
    image = np.zeros((image_height, image_width))

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z = np.array([[xi, yi]])
            x_decoded = model.sample(z)
            digit = tf.reshape(x_decoded[0], (digit_size, digit_size))
            image[i * digit_size: (i + 1) * digit_size,
                    j * digit_size: (j + 1) * digit_size] = digit.numpy()

    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap='Greys_r')
    plt.axis('Off')
    plt.show()
plot_latent_images(model, 20)