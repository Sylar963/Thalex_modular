#!/bin/bash

echo "=========================================="
echo "Launching Thalex Bot & Dashboard"
echo "=========================================="

# Clean up any old processes
echo "Cleaning up old processes..."
pkill -f start_quoter.py
pkill -f streamlit
sleep 2

# Clear old metrics to start fresh
echo "Clearing old metrics..."
rm -f metrics/metrics.csv
rm -f metrics/performance.log

# Start the bot in background
echo "Starting Thalex Bot..."
python start_quoter.py > logs/bot_output.log 2>&1 &
BOT_PID=$!
echo "Bot started with PID: $BOT_PID"

# Wait a few seconds for bot to initialize
echo "Waiting for bot to initialize..."
sleep 10

# Start the dashboard
echo "Starting Dashboard..."
python -m streamlit run dashboard/monitor.py --server.port 8501 --server.address 0.0.0.0 &
DASHBOARD_PID=$!
echo "Dashboard started with PID: $DASHBOARD_PID"

echo ""
echo "=========================================="
echo "🚀 SERVICES LAUNCHED!"
echo "=========================================="
echo "📊 Dashboard: http://localhost:8501"
echo "🤖 Bot PID: $BOT_PID"
echo "📈 Dashboard PID: $DASHBOARD_PID"
echo ""
echo "To stop everything:"
echo "  kill $BOT_PID $DASHBOARD_PID"
echo "  or run: pkill -f start_quoter.py && pkill -f streamlit"
echo ""
echo "Logs:"
echo "  Bot: tail -f logs/bot_output.log"
echo "  Metrics: tail -f metrics/performance.log"
echo "=========================================="

# Wait and show initial status
sleep 5
echo ""
echo "📊 Current Status:"
if ps -p $BOT_PID > /dev/null; then
    echo "✅ Bot is running"
else
    echo "❌ Bot failed to start"
fi

if ps -p $DASHBOARD_PID > /dev/null; then
    echo "✅ Dashboard is running"
else
    echo "❌ Dashboard failed to start"
fi

echo ""
echo "Press Ctrl+C to stop monitoring, or run the kill commands above to stop services"

# Keep script running to monitor
trap "echo 'Stopping services...'; kill $BOT_PID $DASHBOARD_PID 2>/dev/null; exit 0" INT

while true; do
    sleep 30
    if ! ps -p $BOT_PID > /dev/null; then
        echo "⚠️  Bot process died!"
        break
    fi
    if ! ps -p $DASHBOARD_PID > /dev/null; then
        echo "⚠️  Dashboard process died!"
        break
    fi
done 