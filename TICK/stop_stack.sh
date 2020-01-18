#!/bin/bash
command `PID=$(pgrep influxd); sudo kill -9 $PID;`
                        echo -e "\e[1;32m influx stopped! \e[0m";
command `sudo systemctl stop telegraf`
                        echo -e "\e[1;32m telegraf stopped! \e[0m";
command `sudo systemctl stop kapacitor`
                        echo -e "\e[1;32m kapacitor stopped! \e[0m";
command `PID=$(pgrep prometheus); sudo kill -9 $PID;`
                        echo -e "\e[1;32m prometheus stopped! \e[0m";
command `sudo systemctl stop grafana-server`
                        echo -e "\e[1;32m grafana-server stopped! \e[0m";
command `PID=$(pgrep pipeline); sudo kill -9 $PID; sudo pkill screen 2>/dev/null`
                        echo -e "\e[1;32m pipeline stopped! \e[0m";