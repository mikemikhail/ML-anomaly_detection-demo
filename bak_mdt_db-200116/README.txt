$ sudo influxd restore -portable -db mdt_db -newdb mdt_db-200116 .
then change database retention to INF

This is a backup of influxdb database mdt_db. With data for 24 days ending Jan 16th, 2020.
One measurement which is interface generic counters, from 6 XRv9K edge nodes. Including tunnel-te overlay.

To restore https://www.influxdata.com/blog/new-features-in-open-source-backup-and-restore/
may need to specify restore to a new database.
