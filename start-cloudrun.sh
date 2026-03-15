#!/bin/bash
set -e

# Cloud Run injects the PORT env var (default 8080)
PORT="${PORT:-8080}"

echo "Starting HR-VITON on port $PORT"

# Substitute the nginx listen port at runtime
sed -i "s/NGINX_PORT/$PORT/g" /etc/nginx/nginx.conf

# Create required temp directories
mkdir -p /tmp/tryon/uploads /tmp/tryon/outputs
mkdir -p /tmp/nginx_client_body /tmp/nginx_proxy /tmp/nginx_fastcgi /tmp/nginx_uwsgi /tmp/nginx_scgi

# Start all services via supervisord
exec supervisord -c /etc/supervisor/conf.d/supervisord.conf
