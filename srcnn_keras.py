# IMAGE SUPER-RESOLUTION USING SR-CNN
import os
import h5py
from PIL import Image
#import requests
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Activation, UpSampling2D, Add, Average, MaxPooling2D

def generate_patches(img_arr, size=32, stride=16):
  #img_arr.shape --> 0 is #rows, 1 is #columns, 2 is #channels
  i0_list = list(range(0, img_arr.shape[0], stride))
  #print(i0_list)
  j0_list = list(range(0, img_arr.shape[1], stride))
  #print(j0_list)
  patches = np.zeros((len(i0_list) * len(j0_list), size, size, img_arr.shape[2]), dtype=np.float32)
  n = 0
  for i in i0_list:
    for j in j0_list:
      img_patch = img_arr[i:(i+size), j:(j+size), :]
      patches[n, 0:img_patch.shape[0], 0:img_patch.shape[1], 0:img_patch.shape[2]] = img_patch
      n += 1
  return patches / 255


def combine_patches(patches_arr, shape=None, stride=16):
  i0_list = list(range(0, shape[0], stride))
  j0_list = list(range(0, shape[1], stride))
  combined = np.zeros((shape[0], shape[1], patches_arr.shape[3]), dtype=np.float32)
  print('Initial combined shape', combined.shape)
  overlap = np.zeros(shape)
  print('Initial overlap shape',overlap.shape)
  n = 0
  for i in i0_list:
    for j in j0_list:
      if i+patches_arr.shape[1] > shape[0]:
        p_h = shape[0] - i
      else:
        p_h = patches_arr.shape[1]
      if j+patches_arr.shape[2] > shape[1]:
        p_w = shape[1] - j
      else:
        p_w = patches_arr.shape[2]
      patch = patches_arr[n, :p_h, :p_w, :]
      overlap[i:(i+patch.shape[0]), j:(j+patch.shape[1])] += 1
      combined[i:(i+patch.shape[0]), j:(j+patch.shape[1]), :] += patch
      n += 1
  overlap[np.where(overlap == 0)] = 1
  for c in range(combined.shape[2]):
    combined[:, :, c] /= overlap
  combined *= 255
  print('Final Combined shape', combined.shape)
  return np.clip(combined, 0, 255).astype(np.uint8)

# SR-CNN Model Definition
input_shape = (32, 32, 3)

input_layer = Input(shape=input_shape)    
x = Conv2D(64, (9, 9), activation='relu', padding='same', name='level1')(input_layer)
x = Conv2D(32, (1, 1), activation='relu', padding='same', name='level2')(x)
output_layer = Conv2D(3, (5, 5), padding='same', name='output')(x)

model = Model(inputs=input_layer, outputs=output_layer)

model.summary()

#Getting the weights file and assigning the weights to the model.
f = h5py.File(os.path.expanduser('SR Weights 2X.h5'), mode='r')

weights = []
for layer_name in f.attrs['layer_names']:
  g = f[layer_name]
  for weight_name in g.attrs['weight_names']:
    weights.append(g[weight_name].value)

print([w.shape for w in weights])

weights[0] = weights[0].transpose((2,3,1,0))
weights[2] = weights[2].transpose((2,3,1,0))
weights[4] = weights[4].transpose((2,3,1,0))
print([w.shape for w in weights])

model.set_weights(weights)

f.close()

model.save('sr.h5')

#@title Get image from the drive
import cv2
im = cv2.imread('s42_f1_street.bmp',cv2.IMREAD_COLOR)
#im = im[:,:248]
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
print(im.shape)

#@title Take the image, resize + interpolate, run model and show output
plt.imshow(im)
im = Image.fromarray(im)
im = im.resize((im.size[0] * 2, im.size[1] * 2), resample=Image.BILINEAR)
img_arr = np.array(im)

patches = generate_patches(img_arr, size=32, stride=16)
print(patches.shape, patches.dtype)
# Running the CNN model on the test image
result = model.predict(patches)
result = combine_patches(result, shape=img_arr.shape[:2], stride=16)
print('Result shape', result.shape, result.dtype)

fig = plt.figure(figsize=(20,20))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
ax1.imshow(im)
cv2.imwrite('LR_output.jpg',img_arr)
ax2.imshow(result)
cv2.imwrite('Model_output.jpg',result)