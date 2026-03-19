# SuperMesh Project Structure

```
SuperMesh/
├── config/
│   ├── ansible.cfg              # Ansible settings
│   ├── bees.ini                 # Put the  Beelinks and the IPs
│   
│
├── setup/
│   ├── ping.yml                 # test to make sure we talking with all the BeeLinks
│   ├── install_depth_camera.yml # Install depth camera SDK + deps on nodes
│
├── node/                        # Code that runs on each Beelink
│   └── depth_capture.py         # Use the cameras,  send data to central
│
├── central/                     # Code that runs on the central computer
│   └── receiver.py              # Receive data from all 4 nodes (need to figure out what it'll do)
│
├── run_depth_cameras.yml        # Playbook to launch depth_capture.py on all nodes

```
