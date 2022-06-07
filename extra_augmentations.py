import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp
tfd = tfp.distributions
import random

def eraser(image,label):
    p = random.choice([0, 1])
    if(p == 0):
          return image,label

    image = tf.expand_dims(image, axis=0)
    return tf.squeeze(tfa.image.random_cutout(image, (32, 32), constant_values = 1), axis=0),label

def mixup(a, b):
      
  (image1, label1), (image2, label2) = a, b

  alpha = [0.2]
  beta = [0.2]  
  dist = tfd.Beta(alpha, beta)
  l = dist.sample(1)[0][0]
  
  img = l*image1+(1-l)*image2
  lab = l*label1+(1-l)*label2

  return img, lab

def clone (img,label):
  return img,label  