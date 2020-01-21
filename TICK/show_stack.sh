#!/bin/bash

if pgrep -x "influxd" > /dev/null
        then
            echo -e "\e[1;32m InfluxDB is running \e[0m";
        else
            echo -e "\e[1;31m InfluxDB is not running \e[0m";
fi
if pgrep -x "telegraf" > /dev/null
        then
            echo -e "\e[1;32m Telegraf is running \e[0m";
        else
            echo -e "\e[1;31m Telegraf is not running \e[0m";
fi

if pgrep -x "kapacitord" > /dev/null
        then
            echo -e "\e[1;32m Kapacitor is running \e[0m";
        else
            echo -e "\e[1;31m Kapacitor is not running \e[0m";
fi

  if pgrep -x "prometheus" > /dev/null
          then
              echo -e "\e[1;32m Prometheus is running \e[0m";
          else
              echo -e "\e[1;31m Prometheus is not running \e[0m";
  fi
if pgrep -x "grafana-server" > /dev/null
        then
            echo -e "\e[1;32m Grafana is running \e[0m";
        else
            echo -e "\e[1;31m Grafana is not running \e[0m";
fi

 if pgrep -x "pipeline" > /dev/null
         then
             echo -e "\e[1;32m Pipeline is running \e[0m";
         else
             echo -e "\e[1;31m Pipeline is not running \e[0m";
         fi
