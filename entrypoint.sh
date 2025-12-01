#!/bin/bash
set -e

# Fix permissions for mounted volumes
# This runs as root before switching to app user
chown -R app:app /app/logs /app/storage 2>/dev/null || true
chmod -R 755 /app/logs /app/storage 2>/dev/null || true

# Switch to app user and run the application
exec gosu app "$@"

