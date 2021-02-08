#!/bin/bash
rm ./output/uvicorn.log || true
nohup uvicorn server:app --port 9900 --host 0.0.0.0 --reload >> ./output/uvicorn.log 2>&1 &
