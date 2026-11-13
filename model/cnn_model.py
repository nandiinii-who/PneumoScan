
import tensorflow as tf
from keras import layers, models


def build_cnn_model():

    model = models.Sequential([

        # Input image: 224 x 224 pixels, RGB
        layers.Input(shape=(224, 224, 3)),

        # First convolution block
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        # Second convolution block
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        # Third convolution block
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        # Convert feature maps into a vector
        layers.Flatten(),

        # Fully connected layer
        layers.Dense(128, activation="relu"),

        # Reduce overfitting
        layers.Dropout(0.5),

        # Binary classification output
        layers.Dense(1, activation="sigmoid")

    ])

    return model


if __name__ == "__main__":

    model = build_cnn_model()

    model.summary()
