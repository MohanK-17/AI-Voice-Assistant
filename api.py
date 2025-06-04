from fastapi import FastAPI
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from pathlib import Path
import json
import urllib.parse

app = FastAPI()


username = urllib.parse.quote_plus("MohanK-17")
password = urllib.parse.quote_plus("Mohan@2004")

uri = f"mongodb+srv://{username}:{password}@calls.17uvcq5.mongodb.net/?retryWrites=true&w=majority&appName=Calls"
client = MongoClient(uri, server_api=ServerApi('1'))

db = client.calls_db
collection = db.session_logs 

LOG_FILE = Path("speech_log.json")

@app.post("/store-session")
def store_session_to_mongodb():
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        sessions = json.load(f)

    result = collection.insert_many(sessions)
    return {"status": "success", "inserted_ids": [str(_id) for _id in result.inserted_ids]}
