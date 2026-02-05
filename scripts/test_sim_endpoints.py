import requests
import json
import time

BASE_URL = "http://localhost:8000/api/v1"


def test_start_simulation():
    url = f"{BASE_URL}/simulation/start"
    payload = {
        "symbol": "BTC-PERPETUAL",
        "venue": "bybit",
        "start_date": time.time() - 3600,  # Last 1 hour
        "end_date": time.time(),
        "strategy_config": {"risk_aversion": 0.1},
        "risk_config": {"max_position": 10.0},
    }

    print(f"Sending POST request to {url}...")
    print(f"Payload: {json.dumps(payload, indent=2)}")

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        print("\nRequest Successful!")
        print(f"Response: {json.dumps(data, indent=2)}")
        return data.get("run_id")
    except Exception as e:
        print(f"\nRequest Failed: {e}")
        if "response" in locals():
            print(f"Status Code: {response.status_code}")
            print(f"Response Text: {response.text}")
        return None


def get_stats(run_id):
    if not run_id:
        return
    url = f"{BASE_URL}/simulation/{run_id}/stats"
    print(f"\nFetching stats from {url}...")
    try:
        response = requests.get(url)
        print(f"Stats: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Failed to get stats: {e}")


if __name__ == "__main__":
    print("--- Testing Simulation API ---")
    print("Ensure the API server is running (uvicorn src.api.main:app)")

    run_id = test_start_simulation()
    if run_id:
        get_stats(run_id)
