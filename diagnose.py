import os, sys, traceback
from dotenv import load_dotenv
load_dotenv(dotenv_path="/workspaces/airline-support-api/.env", override=True)
sys.path.insert(0, "/workspaces/airline-support-api")

print("=== Env Check ===")
pwd = os.environ.get("DB_PASSWORD", "")
print(f"DB_HOST     = {os.environ.get('DB_HOST')}")
print(f"DB_USER     = {os.environ.get('DB_USER')}")
print(f"DB_PASSWORD = {pwd}  (length: {len(pwd)})")

print("\n=== Testing DB Connection ===")
try:
    import psycopg2
    conn = psycopg2.connect(
        host=os.environ.get("DB_HOST"),
        port=os.environ.get("DB_PORT"),
        user=os.environ.get("DB_USER"),
        password=os.environ.get("DB_PASSWORD"),
        dbname=os.environ.get("DB_NAME")
    )
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM flights")
    print(f"DB OK: {cur.fetchone()[0]} rows")
    cur.close(); conn.close()
except Exception as e:
    print(f"DB Error: {e}")

print("\n=== Testing Full Pipeline ===")
try:
    from app import process_query
    result = process_query("What is the status of flight AI532?")
    print(f"Category : {result['category']}")
    print(f"Response : {result['response'][:200]}")
except Exception as e:
    print(f"Pipeline Error: {e}")
    traceback.print_exc()
