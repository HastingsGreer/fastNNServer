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
    return model.predict(x)

process(np.random.random((512, 512)))
def get_connection():

    s = socket.socket(socket.AF_INET,
    	              socket.SOCK_STREAM)

    s.bind(("", 9999))

    s.listen(1)
    conn, addr = s.accept()
    print("got connection")
    while True:
        message_list = []
        while sum([len(m) for m in message_list]) < 512 * 512:
            message = conn.recv(512 * 512)
            if len(message) == 0:
                print("connection died")
                return
            message_list.append(message)
        message = b"".join(message_list)

        x = np.fromstring(message, dtype=np.uint8).reshape((512, 512)) / 256.0
        res = process(x)
        conn.send(res.astype(np.float32).tobytes())
    	

    s.close()

while True:
    get_connection()
