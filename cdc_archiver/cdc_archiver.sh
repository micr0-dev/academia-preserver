#!/bin/bash

keyword=$1
base_url="https://archive.cdc.gov/#/results?q="
details_url="https://archive.cdc.gov/#/details?url="

mkdir -p "${keyword}/files"
cd "${keyword}"
wb-manager init "cdc-archive-${keyword}"
wayback --record --live -a --auto-interval 2 > /dev/null 2>&1 &
cd ..

sleep 5
echo "Archiving CDC Keyword ${keyword}"
python cdc_archiver_browser.py --keyword "${keyword}"