# SuperMesh Project Structure

```
SuperMesh/
├── config/
│   ├── ansible.cfg              # Ansible settings 
│   ├── bees.ini                 # Put the  Beelinks and the IPs
│   ├── cameras.yaml             # Where each camera is physically mounted (corner, height, room size)
│
├── setup/
│   ├── ping.yml                 # test to make sure we talking with all the BeeLinks
│
├── node/                        # Code that runs on each Beelink
│   └── depth_capture.py         # Blob detection + sends data to central over OSC
│
├── central/                     # Code that runs on the central computer
│   └── receiver.py              # Listens for blob data from all nodes over OSC, builds object map
│
├── concepts/                    # Proof of concept stuff / things we tested, stuff that worked,
│   │                            #   stuff that didn't
│   ├── depth_motion_detection.py  # motion detection + optical flow PoC (this one works)
│   ├── install_depth_camera.yml   # Tried to install RealSense SDK via Ansible (didn't work)
│
├── run_depth_cameras.yml        # Playbook to launch depth_capture.py on all nodes

```
