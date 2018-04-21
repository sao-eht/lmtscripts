# simple test of fill factor on one scan [04.2018 LLB]
length=${1:-'10'}
echo "record=on:$length:$length:fill:test:Lm" | da-client
sleep $length
sleep 5 # extra to let dplane log write out
cat /var/log/mark6/dplane-daemon.log | grep 'packets good' | tail -2
