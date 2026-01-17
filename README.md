# berlin-bikes-big-data
Project made for Data storage in Big Data Systems classes at Warsaw University of Technology

## Setup
1. Start the virtual machine in Oracle Virtualbox
2. Run MobaXterm and start the vm session. Use the key file for authentication
3. Run 
```
sudo /vagrant/scripts/bootstrap.sh
```
4. Run
```
cd /usr/local/kafka
bin/kafka-server-start.sh -daemon config/server.properties
jps
```
you should see Kafka there
5. In the same directory run
```
bin/kafka-topics.sh --bootstrap-server localhost:9092 --create --topic raw.bike --partitions 1 --replication-factor 1
bin/kafka-topics.sh --bootstrap-server localhost:9092 --create --topic raw.weather --partitions 1 --replication-factor 1
bin/kafka-topics.sh --bootstrap-server localhost:9092 --create --topic raw.bvg --partitions 1 --replication-factor 1
```
6. Restart NiFi using:
```
cd /usr/local/nifi-1.14.0/bin
./nifi.sh status
./nifi.sh stop
./nifi.sh start
./nifi.sh status
```
7. Open NiFi in your browser. Upload the template and run everything
8. Check /user/vagrant/nifi_out in hdfs if files are uploading
9. Put spark_silver_merge_1.py and spark_gold.py in /home/vagrant directory
10. Remove checkpoints
```
cd /home/vagrant
hdfs dfs -rm -r -f /chk/silver/bvg_events
hdfs dfs -rm -r -f /chk/silver/weather_observations
hdfs dfs -rm -r -f /chk/silver/nextbike_states
```
11. Run spark silver with Kafka
```
/usr/local/spark/bin/spark-submit \
  --master local[2] \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.1.2 \
  spark_silver_merge_1.py
```
12. Use beeline to see if data are uploading to Hive
```
beeline -u "jdbc:hive2://localhost:10000/" -n vagrant -p vagrant
```
```
SHOW DATABASES;
USE silver;
SHOW TABLES;
SELECT COUNT(*) FROM nextbike_states;
SELECT COUNT(*) FROM weather_observations;
SELECT COUNT(*) FROM bvg_events;
!q
```
13. Run spark_gold.py
```
/usr/local/spark/bin/spark-submit \
  --master local[2] \
  spark_gold.py
```
14. Make a tunnel in MobaXterm. Specifically go to Tunneling -> New SSH tunnel and there type:

Forwarded port: 10000

Remote server: 127.0.0.1

Remote port: 10000

SSH server: all the data that are used to access the vm

15. Start a Python environment in your IDE with requirements.txt provided 
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
16. Run the jupyter notebook
