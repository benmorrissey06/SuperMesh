#Use OSC to message from nodes to central

#implement blob detection code into here

from pythonosc import udp_client
import json
client = udp_client.SimpleUDPClient("CENTRAL_IP_HERE", 9001)

# when we have blobs, send them:
# client.send_message("/blobs/node_id", json.dumps(blobs))