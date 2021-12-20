#!/bin/bash
while true; do
  filename="$(date +%Y%m%d-%H%M%S).pb"
  curl -X GET -H "Content-type: application/octet-stream" -H "Authorization: ${OTD_KEY}" -o "$filename" https://api.opentransportdata.swiss/gtfsrt2020
  echo "$i" "$filename"
  curl --insecure https://localhost:8080/ris/read
  sleep 30
done
