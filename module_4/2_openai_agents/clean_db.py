import sqlite3

DB_PATH = "conversations.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Drop table
cursor.execute("DROP TABLE IF EXISTS faqs")

conn.commit()
conn.close()

print("'faqs' table dropped. You can recreate it now.")
