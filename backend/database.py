import os
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError

# Load environment variables from .env file
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017/")
db_available = False

try:
    # 5 seconds timeout to check if MongoDB is alive.
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.server_info()  # triggers actual connection attempt
    db = client["smartrec"]
    titles_col = db["titles"]
    users_col = db["users"]
    watchlists_col = db["watchlists"]
    watched_col = db["watched"]
    reviews_col = db["reviews"]
    db_available = True
    print(f">> Connected to MongoDB at {MONGO_URI} (db: smartrec)")
except (ServerSelectionTimeoutError, Exception) as e:
    db = None
    titles_col = None
    users_col = None
    watchlists_col = None
    watched_col = None
    reviews_col = None
    db_available = False
    print(f">> [WARN] MongoDB not reachable at {MONGO_URI}. Server will start in fail-safe LOCAL storage fallback mode.")
