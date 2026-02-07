# Imports should work natively now via pip install -e .
import asyncio
import os

from src.adapters.storage.timescale_adapter import TimescaleDBAdapter
from src.domain.entities import Balance, Position


async def verify_balance_storage():
    print("Starting Balance Storage Verification...")

    # 1. Setup DB Connection
    db_user = os.getenv("DATABASE_USER", "postgres")
    db_pass = os.getenv("DATABASE_PASSWORD", "password")
    db_host = os.getenv("DATABASE_HOST", "localhost")
    db_port = os.getenv("DATABASE_PORT", "5433")
    db_name = os.getenv("DATABASE_NAME", "thalex_trading")

    dsn = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
    print(f"Connecting to {db_host}:{db_port}/{db_name}...")

    adapter = TimescaleDBAdapter(dsn)
    try:
        await adapter.connect()
        print("Connected.")
    except Exception as e:
        print(f"Failed to connect: {e}")
        return

    # 2. Save Dummy Balance
    test_exchange = "verify_test"
    test_asset = "USD"
    dummy_balance = Balance(
        exchange=test_exchange,
        asset=test_asset,
        total=1234.56,
        available=1000.00,
        margin_used=234.56,
        equity=1234.56,
    )

    print(f"Saving balance: {dummy_balance}")
    try:
        # Also need to ensure table exists?
        # _init_schema should have run on connect.
        # But we also have _init_balances_table locally which might not be called?
        # Let's check if the table exists by just running save.

        # NOTE: save_balance calls `_init_balances_table`? No, I saw it in the file view,
        # but save_balance logic (853) just does INSERT.
        # Wait, I saw `_init_balances_table` at line 838 but is it CALLED?
        # _init_schema is at line 39. Let's assume it IS called or table exists.

        await adapter.save_balance(dummy_balance)
        print("Save completed (no error raised).")
    except Exception as e:
        print(f"Save failed: {e}")
        # Try finding if table is missing
        if 'relation "account_balances" does not exist' in str(e):
            print("Table account_balances missing! Schema init issue.")

    # 3. Retrieve and Verify
    print("Retrieving latest balances...")
    try:
        balances = await adapter.get_latest_balances()
        found = False
        for b in balances:
            print(f"Found: {b}")
            if b.exchange == test_exchange and b.asset == test_asset:
                if b.total == 1234.56:
                    print("SUCCESS: Balance verified!")
                    found = True
                else:
                    print(f"FAILURE: Value mismatch. Expected 1234.56, got {b.total}")

        if not found:
            print("FAILURE: Test balance not found in retrieval.")

    except Exception as e:
        print(f"Retrieval failed: {e}")

    await adapter.disconnect()


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    asyncio.run(verify_balance_storage())
