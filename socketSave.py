import numpy as np
import socket
import time
import pickle
usScan = []


s = socket.socket(socket.AF_INET,
	              socket.SOCK_STREAM)
try:
	s.bind(("", 9999))

	s.listen(1)
	conn, addr = s.accept()
	print("got connection")
	t_start = time.time()
	while True:
	    message = conn.recv(512 * 512)

	    x = np.fromstring(message, dtype=np.uint8).reshape((512, 512)) / 256.0
	    usScan.append((time.time(), x))		

	s.close()
finally:
	pickle.dump(usScan, open("outputs/" + str(t_start) + "_" + str(time.time()), "wb"))