import inspect
import os
from tensorflow import gfile
import numpy as np
import tensorflow as tf
import time
from tensorflow import logging

out_file = gfile.Open("gs://ksh_imagenet/ILSVRC/feature.csv", "w+")
out_file.write("filename," + ",".join(["feature" + str(i) for i in range(1, 2049)]) + "\n")
files = tf.gfile.Glob("gs://ksh_imagenet/ILSVRC/Data/DET/test/*.JPEG")
logging.info(files)
f=tf.gfile.FastGFile("gs://ksh_imagenet/vgg16/classify_image_graph_def.pb", 'rb')
graph_def = tf.GraphDef()
graph_def.ParseFromString(f.read())
_ = tf.import_graph_def(graph_def, name='')

config = tf.ConfigProto(allow_soft_placement = True)
sess = tf.Session(config = config)
for k in range(1):
    with tf.device("/gpu:%d" % k):
        pool_3_tensor = sess.graph.get_tensor_by_name('pool_3:0')

        for i in range(len(files)):
            image_data = tf.gfile.FastGFile(files[i], 'rb').read()

            logging.info(i)
            feature = sess.run(pool_3_tensor,
                                   {'DecodeJpeg/contents:0': image_data})
            feature = np.squeeze(feature)
            out_file.write(files[i] + "," + ",".join(["%f" % y for y in feature]) + "\n")
            
            
out_file.close()

