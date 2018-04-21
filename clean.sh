sudo /etc/init.d/cplane stop
sudo /etc/init.d/dplane stop
sudo /etc/init.d/dplane start
sleep 5
sudo /etc/init.d/cplane start
