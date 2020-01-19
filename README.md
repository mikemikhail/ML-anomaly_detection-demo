# ML-anomaly_detection-demo

This Repo is to accompany the TECDEV-2765 Cisco Live Europe session
it includes code to follow along with the presenters

You can save time by running these ahead of time to download the vagrant image and containers to beforehand

to start up the Collection Stack, simply type "vagrant up"
to connect to the VM, type "vagrant ssh TICK"

to enter the tensorflow python virtualenv type "source ~/TF1/bin/activate"

to start up the jupyter notebook to query the stack type 
"docker run -p 8888:8888 --rm --privileged -v$(pwd):/home/dockerkuhlskev/jupter_ml:1.1"

to start up the container in the shell and run code manually type 
"docker run -it --rm --privileged -v$(pwd):/home/docker kuhlskev/jupter_ml:1.1 /bin/sh"
