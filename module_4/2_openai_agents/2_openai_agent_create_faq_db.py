import sqlite3

# Define the path to the database file
DB_PATH = "c://code//agenticai//2_openai_agents//faqs.db"

# Connect to SQLite (creates the file if it doesn't exist)
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Create the 'faqs' table
cursor.execute("""
CREATE TABLE IF NOT EXISTS faqs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic TEXT NOT NULL,
    answer TEXT NOT NULL
)
""")

# Example FAQ entries
faq_entries = [
    ("shipping_time", "Our standard shipping time is 3-5 business days."),
    ("return_policy", "You can return any product within 30 days of delivery."),
    ("warranty", "All products come with a one-year warranty covering manufacturing defects."),
    ("payment_methods", "We accept credit cards, debit cards, and PayPal."),
    ("customer_support", "You can reach our support team 24/7 via email or chat.")
]

# Insert entries into the table
cursor.executemany("""
INSERT INTO faqs (topic, answer)
VALUES (?, ?)
""", faq_entries)

# Commit changes and close connection
conn.commit()
conn.close()

print("Database 'faqs.db' created and populated successfully!")