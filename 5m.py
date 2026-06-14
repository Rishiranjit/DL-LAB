#5 a
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
X = np.array([[1],[2],[3],[4],[5]])
y = np.array([2,4,6,8,10])
model1 = LinearRegression()
model1.fit(X,y)
model2 = Ridge(alpha=0.1)
model2.fit(X,y)
print("Without Regularization")
print("Weight:", model1.coef_[0])
print("Bias:", model1.intercept_)
print("\nWith L2 Regularization")
print("Weight:", model2.coef_[0])
print("Bias:", model2.intercept_)

#5 b
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
img = image.load_img("cat1.png", target_size=(150,150))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
i = 0
for batch in datagen.flow(img, batch_size=1):
    plt.subplot(2,3,i+1)
    plt.imshow(batch[0].astype("uint8"))
    plt.axis("off")
    i += 1
    if i == 6:
        break
plt.show()
