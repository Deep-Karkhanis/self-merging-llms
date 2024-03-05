#!/bin/bash

# run this as a cron to automatically update the ssl certs and bounce nginx
# 35 */4 * * * /home/ubuntu/bin/update_ssl_certs.sh 1>/tmp/update_ssl_certs.log 2>&1


cert_loc=gs://realityengines-keys-external/ssl
key_file=reai.io.key
cert_file=reai.io.crt

local_dir=/etc/nginx/ssl

me="$(basename $0)"
function _print {
    now=$(date +"%F %T")
    echo "$now $me : $1"
}

_print "INFO: begin ssl cert refresh check from $cert_loc to $local_dir"

if [[ ! -d $local_dir ]]; then
    _print "setting up $local_dir"
    sudo mkdir -p $local_dir
    sudo chown ubuntu:ubuntu $local_dir
    sudo chmod 750 $local_dir
fi

tmp_dir=/tmp/ssl
mkdir -p $tmp_dir

nginx_reload="no"

for file in $key_file $cert_file; do
    gsutil -q cp $cert_loc/$file $tmp_dir/
    if [[ $? == 0 ]]; then
        diff $local_dir/$file $tmp_dir/$file 1>/dev/null 2>&1
        if [[ $? != 0 ]]; then
            _print "INFO: updating $file"
            cp $tmp_dir/$file $local_dir/
            nginx_reload="yes"
        fi
    else
        _print "ERROR: while downloading $cert_loc/$file"
    fi
    rm -f $tmp_dir/$file
done

if [[ $nginx_reload == "yes" ]]; then
    sudo service nginx reload
    if [[ $? == 0 ]]; then
        _print "INFO: Reloading nginx due to updated ssl certs"
    else
        _print "ERROR: while reloading nginx"
    fi
else
    _print "INFO: ssl certs files have not changed"
fi

_print "INFO: end ssl cert refresh check"
