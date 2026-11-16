#!/usr/bin/env bash
# Download the Rent the Runway fit-feedback dataset (Misra, Wan & McAuley, RecSys 2018).
set -e
mkdir -p data
URL=https://mcauleylab.ucsd.edu/public_datasets/data/renttherunway/renttherunway_final_data.json.gz
curl -L --fail -o data/renttherunway_final_data.json.gz "$URL" || {
  echo "UCSD mirror unavailable, trying a GitHub copy"
  curl -L --fail -o data/renttherunway_final_data.json.gz \
    https://raw.githubusercontent.com/Sriharsha-Pamidi/Multi-Class-Fit-Prediction/main/data/renttherunway_final_data.json.gz
}
echo "saved data/renttherunway_final_data.json.gz"
