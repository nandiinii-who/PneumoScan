
import numpy as np
import tensorflow as tf

def preprocess_image(image):

    # Convert image to RGB
    image = image.convert("RGB")

    # Convert to NumPy array
    image_array = np.array(image)

    # Use the same resizing method as training
    image_tensor = tf.image.resize(
        image_array,
        (224, 224)
    )

    # Normalize pixel values
    image_tensor = tf.cast(
        image_tensor,
        tf.float32
    ) / 255.0

    # Add batch dimension
    image_tensor = tf.expand_dims(
        image_tensor,
        axis=0
    )

    return image_tensor.numpy()
