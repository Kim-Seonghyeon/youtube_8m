import inspect
import os
from tensorflow import gfile
import numpy as np
import tensorflow as tf
import time
from tensorflow import logging


sess = tf.Session()
with tf.device("/gpu:0"):
    f=tf.gfile.FastGFile("gs://ksh_imagenet/vgg16/classify_image_graph_def.pb", 'rb')
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

    softmax_tensor = sess.graph.get_tensor_by_name('pool_3:0')
    files = tf.gfile.Glob("gs://ksh_imagenet/ILSVRC/Data/DET/test/*.JPEG")
    logging.info(files)
    out_file = gfile.Open("gs://ksh_imagenet/ILSVRC/feature.csv", "w+")
    out_file.write("filename," + ",".join(["feature" + str(i) for i in range(1, 2049)]) + "\n")


    for i in range(len(files)):
        image_data = tf.gfile.FastGFile(files[i], 'rb').read()

        logging.info(i)
        feature = sess.run(softmax_tensor,
                               {'DecodeJpeg/contents:0': image_data})
        feature = np.squeeze(feature)
        out_file.write(files[i] + "," + ",".join(["%f" % y for y in feature]) + "\n")
    out_file.close()

