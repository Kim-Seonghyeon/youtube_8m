import inspect
import os
from tensorflow import gfile
import numpy as np
import tensorflow as tf
import time
from tensorflow import logging


logging.info("22")
files = gfile.Glob("gs://ksh_imagenet/ILSVRC/Data/DET/test/ILSVRC2016_test_00000001.JPEG")
logging.info(files)
out_file = gfile.Open("gs://ksh_imagenet/vgg16/feature.csv", "w+")
out_file.write("filename,"+",".join(["feature"+str(i) for i in range(1,4097)])+"\n")
filename_queue = tf.train.string_input_producer(files) #  list of files to read

reader = tf.WholeFileReader()
key, value = reader.read(filename_queue)

my_img = tf.image.decode_jpeg(value) # use png or jpg decoder based on your files.
my_img = tf.image.resize_images(my_img,[224,224]) # use png or jpg decoder based on your files.

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
  sess.run(init_op)

  # Start populating the filename queue.

  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)
  image0 = []
  for i in range(len(files)): #length of your filename list
    image0.append(my_img.eval()) #here is your image Tensor :)

  image0 = np.expand_dims(image0, axis=0)
  image0 = np.squeeze(image0, 0)/255


  coord.request_stop()
  coord.join(threads)

sess= tf.Session()
model= Vgg16()
model.build(image0)
feature=sess.run(model.fc7)
for i in range(len(files)):
    out_file.write(files[i] + "," + ",".join(["%f" % y for y in feature[i]]) + "\n")
out_file.close()






