#!/bin/bash

tmp_filename="$(dirname $0)/tmp/copy_models.txt"

touch "$tmp_filename"

sig_handler() {
    rm "$tmp_filename"
    echo "Removed tmp file $tmp_filename"
}

trap "sig_handler" EXIT SIGINT SIGTERM SIGCONT

python $(dirname $0)/py_scripts/copy_models.py "$@" > "$tmp_filename"

while IFS=, read -r src_name dest_name; do
    echo "$src_name"
    echo "$dest_name"
    rsync -r --info=progress2 "$src_name" "$dest_name"
done < "$tmp_filename"