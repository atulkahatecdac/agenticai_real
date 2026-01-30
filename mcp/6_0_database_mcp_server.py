import sqlite3
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("sqlite-database")

# Create a simple SQLite database
def init_database():
    conn = sqlite3.connect('simple_db.sqlite')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

@mcp.tool()
def add_user(name: str, email: str) -> str:
    """
    Add a new user to the database
    
    Args:
        name: User's name
        email: User's email
    """
    conn = sqlite3.connect('simple_db.sqlite')
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO users (name, email) VALUES (?, ?)",
        (name, email)
    )
    conn.commit()
    conn.close()
    return f"Added user: {name} ({email})"

@mcp.tool()
def list_users() -> str:
    """
    List all users in the database
    """
    conn = sqlite3.connect('simple_db.sqlite')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users")
    users = cursor.fetchall()
    conn.close()
    
    if not users:
        return "No users found"
    
    result = "Users:\n" + "\n".join(
        f"ID: {user[0]}, Name: {user[1]}, Email: {user[2]}" 
        for user in users
    )
    return result

if __name__ == "__main__":
    init_database()
    mcp.run()