#!/bin/bash

# Copy CA cert to trusted store  
cp /home/site/wwwroot/wccrootca.crt /usr/local/share/ca-certificates/wccrootca.crt 
cp /home/site/wwwroot/wccrootca.cer /usr/local/share/ca-certificates/wccrootca.cer 
  
# Update CA certificates store  
update-ca-certificates  

uvicorn backend.api:app --host 0.0.0.0 --port 8000
# gunicorn backend.api:app --bind 0.0.0.0:8000 -k uvicorn.workers.UvicornWorker