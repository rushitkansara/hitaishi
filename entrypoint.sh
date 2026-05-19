#!/bin/sh
# Hardcode port to 8000 to bypass interpolation issues
uvicorn api.main:app --host 0.0.0.0 --port 8000
