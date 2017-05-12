import inspect
import os
from tensorflow import gfile
import numpy as np
import tensorflow as tf
import time
from tensorflow import logging


f=tf.gfile.FastGFile("gs://ksh_imagenet/vgg16/classify_image_graph_def.pb", 'rb')
graph_def = tf.GraphDef()
graph_def.ParseFromString(f.read())
_ = tf.import_graph_def(graph_def, name='')
sess = tf.Session()

image_data = tf.gfile.FastGFile("gs://ksh_imagenet/ILSVRC/Data/DET/test/ILSVRC2016_test_00000001.JPEG", 'rb').read()


softmax_tensor = sess.graph.get_tensor_by_name('pool_3:0')
predictions = sess.run(softmax_tensor,
                       {'DecodeJpeg/contents:0': image_data})
predictions = np.squeeze(predictions)
logging.info(predictions.shape)
logging.info(predictions)
print(predictions.shape)
print(predictions)






