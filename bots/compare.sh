#!/bin/bash

read -p "How many times do you want to run the comparison? " count

for ((i=1; i<count; i++)); do
  echo "Round $i:"
  luxai-s3 ai/main.py example/main.py
done
