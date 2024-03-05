#!/bin/bash

# script to setup local iptables restrictions on lambda hosts

aws_ips='35.171.48.46 54.237.164.35 23.21.205.49 18.210.122.144 3.133.213.104 18.219.219.10 3.13.149.248 3.130.176.57 3.134.202.125 18.188.43.170 3.18.180.86 34.200.209.129 34.241.157.186 3.99.255.130 54.186.208.74 13.232.237.130 3.221.3.4 52.23.111.68 18.219.219.10 3.134.202.125 3.17.99.211'
gcp_ips='104.197.216.98 104.197.9.141 35.188.191.221 34.29.120.158 34.133.243.145 34.128.79.26'
azure_ips='20.22.31.127 20.232.69.8'
admin_ips=''

ufw default deny incoming
ufw default allow outgoing

ufw allow in on tailscale0

for ip in $aws_ips $gcp_ips $azure_ips $admin_ips; do
    ufw allow from $ip
done

ufw --force enable 
