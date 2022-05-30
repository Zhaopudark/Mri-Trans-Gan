import functools

import numpy as np
import tensorflow as tf

INPUT_SHAPE = (32, 32, 3)
BATCH_SIZE = 256
CLASS_COUNT = 10


def get_train_val_test():
    width, height, _ = INPUT_SHAPE

    get_dataset = functools.partial(
        tf.keras.utils.image_dataset_from_directory,
        batch_size=BATCH_SIZE,
        image_size=(width, height),
        label_mode="categorical",
        shuffle=True,
        seed=42,
        color_mode="rgb",
    )

    train: tf.data.Dataset = get_dataset(directory="/dev/shm/dataset/train")
    val: tf.data.Dataset = get_dataset(directory="/dev/shm/dataset/val")
    test: tf.data.Dataset = get_dataset(directory="/dev/shm/dataset/test")

    # sanity checks
    train_data_sample, train_target_sample = list(train.take(1).as_numpy_iterator())[0]
    assert train_data_sample.shape == (BATCH_SIZE, *INPUT_SHAPE)
    assert train_target_sample.shape == (BATCH_SIZE, CLASS_COUNT)

    val_data_sample, val_target_sample = list(val.take(1).as_numpy_iterator())[0]
    assert val_data_sample.shape == (BATCH_SIZE, *INPUT_SHAPE)
    assert val_target_sample.shape == (BATCH_SIZE, CLASS_COUNT)

    test_data_sample, test_target_sample = list(test.take(1).as_numpy_iterator())[0]
    assert test_data_sample.shape == (BATCH_SIZE, *INPUT_SHAPE)
    assert test_target_sample.shape == (BATCH_SIZE, CLASS_COUNT)

    return (
        train.prefetch(128),
        val.prefetch(128),
        test.prefetch(128),
    )


def get_model():
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=INPUT_SHAPE),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, strides=2, padding="same"
            ),
            tf.keras.layers.Activation("relu"),
            #
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, strides=2, padding="same"
            ),
            tf.keras.layers.Activation("relu"),
            #
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128),
            tf.keras.layers.Activation("relu"),
            #
            tf.keras.layers.Dense(CLASS_COUNT),
            tf.keras.layers.Activation("softmax"),
        ]
    )

    model.compile(
        optimizer="nadam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def why_u_inconsistent():
    tf.keras.backend.clear_session()
    tf.random.set_seed(42)
    np.random.seed(42)

    train, val, test = get_train_val_test()
    model = get_model()
    model.fit(train, validation_data=val, epochs=5, verbose=0)
    scores = model.evaluate(test, verbose=0)
    for name, value in zip(model.metrics_names, scores):
        print(f"test {name}: {value}")


def main():
    for run_nr in range(2):
        print(f"run: {run_nr}")
        why_u_inconsistent()
        print("=====")


if __name__ == "__main__":
    main()