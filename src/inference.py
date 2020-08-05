import numpy as np

# pylint: disable=E0401, W0611
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import img_to_array
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from PIL import Image

WEIGHTS = './tensorfood.h5'
FOODS = ['chilli_crab',
         'curry_puff',
         'dim_sum',
         'ice_kacang',
         'kaya_toast',
         'nasi_ayam',
         'popiah',
         'roti_prata',
         'sambal_stingray',
         'satay',
         'tau_huay',
         'wanton_noodle']


class Inference:

    def __init__(self, shape=224, lr=1e-4):
        self.model = None
        self.image = None
        self.shape = shape
        self.lr = lr

    def make_inference(self, image):
        """Main inference function to make predictions on given image

        Arguments:
            image {.jpg, .jpeg, .png} -- Takes in an image

        Returns:
            preds {dict} -- Dictionary containing label and probability
        """
        # Load image for inference
        self.image = self._load_img(image)

        # Make predictions
        preds = self.model.predict(self.image)

        return FOODS, preds

    def load_model(self):
        """Loads our model (including architecture and weights)

        Returns:
            model -- tensorflow model object
        """
        # Use a base model + some layers
        self.model = self._get_base_model()
        self.model = self._build_model()

        # Add optimizer and compile
        optimizer = SGD(lr=self.lr, momentum=0.9)
        self.model.load_weights(WEIGHTS)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=optimizer,
                           metrics=['accuracy'])

        return self.model

    def _build_model(self):
        """Builds our architecture based on the weights file we are using

        Returns:
            model -- tensorflow model object
        """
        x = self.model.output
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.0)(x)
        x = Flatten()(x)
        x = Dense(12, activation='softmax')(x)
        self.model = tf.keras.Model(inputs=self.model.input, outputs=x)

        return self.model

    def _get_base_model(self):
        """Loads ResNet50 architecture

        Returns:
            model -- tensorflow model object
        """
        model = ResNet50(
            input_shape=(self.shape, self.shape, 3),
            include_top=False,
            weights=None)

        return model

    def _load_img(self, image):
        """Load and resize image

        Arguments:
            image {.jpg, .jpeg, .png} -- Image in the relevant format

        Returns:
            resized_img -- Tensored image
        """
        rgb_img = self._convert_rgb(image)
        resized_img = self._resize_image(rgb_img)
        resized_img = np.expand_dims(resized_img, axis=0)

        return resized_img

    def _resize_image(self, image):
        """Resize image

        Arguments:
            image {.jpg, .jpeg, .png} -- Image in PIL form

        Returns:
            new_im -- Tensored image
        """
        size = self.shape, self.shape
        new_im = image.resize(size)
        new_im = img_to_array(new_im)
        new_im = img_to_array(new_im)

        return new_im

    def _convert_rgb(self, image):
        """Convert image to 3 channels

        Arguments:
            image {.jpg, .jpeg, .png} -- Image in the relevant format

        Returns:
            Image in PIL format
        """
        return Image.open(image).convert('RGB')


if __name__ == "__main__":
    parser = ArgumentParser(description="Takes in an image to make inference",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image", type=str, default="test_images/test1.jpg")

    # Consolidate arguments
    args = parser.parse_args()
    params = {
        'image': args.image
    }

    inf = Inference()
    inf.load_model()
    inf.make_inference(params['image'])
