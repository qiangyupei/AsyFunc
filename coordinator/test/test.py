import json
import time
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

def openPort(port):
    try:
        # Create a socket object
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        # Bind the socket to a specific address and port
        s.bind(('localhost', port))
        
        # Listen for incoming connections
        s.listen(1)
        
        print(f"Port {port} is open")
        while True:
            # Accept incoming connections
            conn, addr = s.accept()
            
            # Receive data from the client
            data = conn.recv(1024)
            
            print(f"Received data on port {port}: {data.decode()}")
            
            # Close the connection
            conn.close()
        
    except socket.error as e:
        print(f"Port {port} is closed: {e}")
        
    finally:
        # Close the socket
        s.close()


class MyRequestHandler(BaseHTTPRequestHandler):
    # Handle POST requests
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length).decode('utf-8')
        print(post_data)
        try:  
            json_data = json.loads(post_data)  
            print("Received JSON data:", json_data)  
        except json.JSONDecodeError as e:  
            print("Invalid JSON data received")
            print(e.msg)  

        # Send response status code
        self.send_response(200)
        self.headers['Content-type'] = 'application/json'
        self.end_headers()
        self.wfile.write(json.dumps({'status': 'ok'}).encode())

if __name__ == "__main__":

    PORT = 8080
    # Create the HTTP server
    httpd = HTTPServer(('127.0.0.1', PORT), MyRequestHandler)
    print("Server started at 127.0.0.1:" + str(PORT))
    # Start the HTTP server
    thread0 = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread0.start()

    # Create new threads and start socket server
    thread1 = threading.Thread(target=openPort, args=(8898,), daemon=True)
    thread1.start()

    thread2 = threading.Thread(target=openPort, args=(8899,), daemon=True)
    thread2.start()

    bodyInstance = {'type': 'body', 'name': 'body1', 'IP': '127.0.0.1'}
    shadowInstance = {'type': 'shadow', 'name': 'shadow1', 'IP': '127.0.0.1', 'body': 'body1'}
    shadowInstance['layers'] = ['L1', 'L2', 'L3']
    time.sleep(3)

    ''' Instance Registration Test '''
    with open('/dev/shm/AsyFunc-Registry-body1.json', 'w') as f:
        json.dump(bodyInstance, f)
    with open('/dev/shm/AsyFunc-Registry-shadow1.json', 'w') as f:
        json.dump(shadowInstance, f)
    
    time.sleep(3)
    ''' Instance Communication Test '''
    with open('/dev/shm/AsyFunc-Body-body1-layer1.json', 'w') as f:
        json.dump({}, f)
    with open('/dev/shm/AsyFunc-Shadow-shadow1-layer1.json', 'w') as f:
        json.dump({}, f)
    time.sleep(3)
    ''' Instance Delete Test '''
    with open('/dev/shm/AsyFunc-Delete-shadow1.json', 'w') as f:
        json.dump({'type': 'shadow', 'name': 'shadow1'}, f)
    with open('/dev/shm/AsyFunc-Delete-body1.json', 'w') as f:
        json.dump({'type': 'body', 'name': 'body1'}, f)

    time.sleep(20)