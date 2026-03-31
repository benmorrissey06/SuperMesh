#Use OSC to message from nodes to central

from pythonosc import dispatcher, osc_server

def handle_blobs(address, payload):
    print(address, payload)

disp = dispatcher.Dispatcher()
disp.map("/blobs/*", handle_blobs)

server = osc_server.ThreadingOSCUDPServer(("0.0.0.0", 9001), disp)
server.serve_forever()
