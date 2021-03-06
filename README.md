# ML-anomaly_detection-demo

This Repo is to accompany the TECDEV-2765 Cisco Live Europe session
it includes code to follow along with the presenters

Requrements to run this locally are Virtualbox (we used 6.0), Vagrant, X-Server (used Quartz on Mac), and Git

First clone the repo with 

$ git clone https://github.com/mikemikhail/ML-anomaly_detection-demo 

You can save time by running this command ahead of time to download the vagrant image beforehand
to start up the Collection Stack, simply switch to clone directory type "vagrant up"

$ cd ~/ML-anomaly_detection-demo/

$ vagrant up

Once you have the VM up and returns to your prompt you ready to move on 

to connect and see the generated ML graphs via x-server

$ ssh -Y -p 2222 -i ~/.vagrant.d/insecure_private_key vagrant@127.0.0.1

or go to virtual machine window in Virtualbox, login vagrant/vagrant, and open up a terminal

to start up the telemetry stack 

$ sh tecdev-2765/TICK/start_stack.sh

$ start influxdb &

Once that is up you can run the ML script with the following command
python tecdev-2765/monitor.py

--------
Optional:
From your laptop you can connect to grafana and see the data visualizations

http://localhost:3000
creds are admin/admin

To visualize data from Dec 24, 2019 to Jan 16, 2020:

in grafana go to setting and change InfluxDB database to mdt_db-200116
and see in dashboard "monitored"
![Grafana monitored dashboard](https://github.com/mikemikhail/ML-anomaly_detection-demo/blob/master/demo-dashboard.png)
--------

To start Machine Learning prediction & Anomaly Detection:

$ cd tecdev-2765

to remove any previous, stopped model and create a fresh one on first learning cycle:

$ rm -r model/

$ python moniotr.py

In a few seconds you'll see descriptions of large dataframes of numbers. After the first large training cycle, you'll see plots. Initially, there's 3 lage learning activities, 5 minutes apart, with plots updated after each. Then smaller data learn/predict/compare every 10 minutes. The new model will improve with experience!

![3 plots](https://github.com/mikemikhail/ML-anomaly_detection-demo/blob/master/demo.png)

A model is born!

The program takes you to a start on same time of day on January 9th, 2020. From there can run until data is exhausted, about a week to last data of January 16th, 2020. Previous period data is used for training and prediction.
