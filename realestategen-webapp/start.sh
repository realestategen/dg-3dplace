#!/bin/bash
# Start backend and frontend dev servers
BASE="$(cd "$(dirname "$0")" && pwd)"

echo "Starting backend on port 8765..."
cd "$BASE/backend"
.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8765 --reload &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"

echo "Starting frontend on port 5174..."
cd "$BASE/frontend"
npm run dev -- --port 5174 &
FRONTEND_PID=$!
echo "Frontend PID: $FRONTEND_PID"

echo ""
echo "  Backend:  http://localhost:8765"
echo "  Frontend: http://localhost:5174"
echo "  API docs: http://localhost:8765/docs"
echo ""
echo "Press Ctrl+C to stop both servers."

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null" EXIT
wait
