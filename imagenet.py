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
softmax_tensor = sess.graph.get_tensor_by_name('pool_3:0')
files = tf.gfile.Glob("gs://ksh_imagenet/ILSVRC/Data/DET/test/*.jpeg")
out_file = gfile.Open("gs://ksh_imagenet/ILSVRC/feature.csv", "w+")
out_file.write("filename," + ",".join(["feature" + str(i) for i in range(1, 2049)]) + "\n")


for i in range(len(files)):
    image_data = tf.gfile.FastGFile(files[i], 'rb').read()

    # 몇가지 유용한 텐서들:
    # 'softmax:0': 1000개의 레이블에 대한 정규화된 예측결과값(normalized prediction)을 포함하고 있는 텐서
    # 'pool_3:0': 2048개의 이미지에 대한 float 묘사를 포함하고 있는 next-to-last layer를 포함하고 있는 텐서
    # 'DecodeJpeg/contents:0': 제공된 이미지의 JPEG 인코딩 문자를 포함하고 있는 텐서

    # image_data를 인풋으로 graph에 집어넣고 softmax tesnor를 실행한다.
    feature = sess.run(softmax_tensor,
                           {'DecodeJpeg/contents:0': image_data})
    feature = np.squeeze(feature)
    out_file.write(files[i] + "," + ",".join(["%f" % y for y in feature]) + "\n")
out_file.close()
