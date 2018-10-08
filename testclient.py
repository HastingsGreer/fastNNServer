import socket

host = socket.gethostname()    
port = 9999             # The same port as used by the server
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((host, port))
s.sendall(bytes(range(256)))

