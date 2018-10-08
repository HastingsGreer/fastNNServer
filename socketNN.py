
import scipy
import numpy as np
import socket
import matplotlib.pyplot as plt
import keras

model = keras.models.load_model("../spineAngle/interson_human_14")


def process(x):
    
    x = scipy.misc.imresize(x, (128, 128)) / 255.
    x = np.expand_dims(x, 0)
    x = np.expand_dims(x, -1)
    

    print(model.predict(x))

process(np.random.random((512, 512)))
s = socket.socket(socket.AF_INET,
	              socket.SOCK_STREAM)

s.bind(("", 9999))

s.listen(1)
conn, addr = s.accept()
print("got connection")
while True:
    message = conn.recv(512 * 512)

    x = np.fromstring(message, dtype=np.uint8).reshape((512, 512)) / 256.0
    process(x)
	

s.close()