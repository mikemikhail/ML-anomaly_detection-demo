#!/bin/bash

command `sudo systemctl start telegraf`
                        echo -e "\e[1;32m Telegraf Done! \e[0m";
command `sudo systemctl start kapacitor`
                        echo -e "\e[1;32m Kapacitor Done! \e[0m";
command `sudo systemctl start grafana-server`
                        echo -e "\e[1;32m Grafana Done! \e[0m";
#command `sudo influxd -config /etc/influxdb/influxdb.conf &`
#                        echo -e "\e[1;32m Influx Done! \e[0m";
#command `cd ~/analytics/prometheus/prometheus-1.5.2.linux-amd64; sudo ~/analytics/prometheus/prometheus-1.5.2.linux-amd64/prometheus &`
#                        echo -e "\e[1;32m Prometheus Done! \e[0m";
#command `screen -dm -S Pipeline bash -c 'cd ~/analytics/pipeline; sudo ./bin/pipeline -config pipeline.conf -pem ~/.ssh/id_rsa; exec /bin/bash &'`
#                        echo -e "\e[1;32m Pipeline Done! \e[0m";