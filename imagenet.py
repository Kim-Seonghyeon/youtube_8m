
from tensorflow import gfile
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import numpy as np

base_model = VGG19(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

batch_size=10
reach_last_file=False
idx = np.array([0,batch_size])

files = gfile.Glob("C:/Users/Administrator/Downloads/ILSVRC2016_DET_test_new.tar/ILSVRC/Data/DET/test/*.jpeg")
out_file = gfile.Open("C:/Users/Public/Pictures/Sample Pictures/aa.csv", "w+")

out_file.write("filename,"+",".join(["feature"+str(i) for i in range(1,4097)])+"\n")
while not reach_last_file:
    if idx[1] == len(files):
        reach_last_file = True

    files0=files[idx[0]:idx[1]]
    imgs = [image.load_img(files0[i], target_size=(224, 224)) for i in range(len(files0))]
    x = [image.img_to_array(img) for img in imgs]
    x = np.expand_dims(x, axis=0)
    x = np.squeeze(x,0)
    x = preprocess_input(x)

    block4_pool_features = model.predict(x)
    for i in range(len(files0)):
        out_file.write(files0[i]+","+ ",".join(["%f" % i for i in block4_pool_features[i]]) + "\n")

    idx += batch_size
    idx[1] = min([idx[1],len(files)])
out_file.close()
