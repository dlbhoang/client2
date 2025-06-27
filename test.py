from pymongo import MongoClient

client = MongoClient("mongodb+srv://admin:admin@app.sj5nx.mongodb.net/?retryWrites=true&w=majority&appName=app")

print("📂 Danh sách database:")
print(client.list_database_names())

db = client["test"]
print("\n📄 Danh sách collection trong DB 'app':")
print(db.list_collection_names())
