import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

img=image.load_img("img1.jpeg",target_size=(200,200))
img=image.img_to_array(img)
img=np.expand_dims(img,0)

datagen=ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

titles=["Rotation","Width Shift","Height Shift",
        "Zoom","Horizontal Flip","Augmented"]

aug=datagen.flow(img,batch_size=1)

for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(next(aug)[0].astype("uint8"))
    plt.title(titles[i])
    plt.axis("off")

plt.show()
