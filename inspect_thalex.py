import inspect

try:
    import thalex.thalex as tx

    print("Thalex module found.")
    client = tx.Thalex()
    print("Client methods:")
    for name, method in inspect.getmembers(client):
        if not name.startswith("_"):
            print(f" - {name}")

    # Check if network class exists
    print(f"\nNetwork Enum: {tx.Network}")
except ImportError:
    print("Thalex module NOT found.")
except Exception as e:
    print(f"Error: {e}")
