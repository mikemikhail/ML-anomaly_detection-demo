# ML-anomaly_detection-demo

This Repo is to accompany the TECDEV-2765 Cisco Live Europe session
it includes code to follow along with the presenters

Requrements to run this locally are Virtualbox (we used 6.0), Vagrant, X-Server (used Quartz on Mac), and Git

First clone the repo with 

git clone https://github.com/mikemikhail/ML-anomaly-detection-demo.git

You can save time by running this command ahead of time to download the vagrant image beforehand
to start up the Collection Stack, simply type "vagrant up"

$ vagrant up

Once you have the VM up and returns to your prompt you ready to move on 

to connect to the VM, type

$ vagrant ssh TICK

or to connect and see the generated ML graphs via x-server

$ ssh -Y -p 2222 -i ~/.vagrant.d/insecure_private_key vagrant@127.0.0.1

to start up the telemetry stack 

$ sh /tecdev-2765/TICK/start_stack.sh

$ start influxdb &

Once that is up you can run the ML script with the following command
python tecdev-2765/monitor.py

From your laptop you can connect to grafana and see the data visualizations

http://localhost:3000
creds are admin/admin



