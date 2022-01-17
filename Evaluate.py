import tensorflow as tf
import numpy as np

IMG_HEIGHT = 256
IMG_WIDTH = 256


# load image
test_dir = "test/pass1.png"
test_image = tf.keras.utils.load_img(test_dir, target_size=(IMG_HEIGHT, IMG_WIDTH))
img_array = tf.keras.utils.img_to_array(test_image)
img_array=img_array/255
img_array = tf.expand_dims(img_array, 0) # Create a batch

# Load trained model
model = tf.keras.models.load_model('Trained_model/model')

# add softmax layer
model.add(tf.keras.layers.Softmax())
prediction = model.predict(img_array)

# 1: Fail 2: Pass
print("---------Classificaiton report--------")
print("Class confidence: ",prediction[0])
if np.argmax(prediction[0])==1:
    print("Passed")
else:
    print("Failed")


