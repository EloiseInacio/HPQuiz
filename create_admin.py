"""
create_admin.py — Bootstrap the first admin account.

Usage:
    python create_admin.py <username> <password>
"""

import os
import sqlite3
import sys

from werkzeug.security import generate_password_hash

USERS_DB_PATH = os.environ.get("HPQUIZ_USERS_DB", "users.db")


def main():
    if len(sys.argv) != 3:
        print("Usage: python create_admin.py <username> <password>")
        sys.exit(1)

    username, password = sys.argv[1], sys.argv[2]
    conn = sqlite3.connect(USERS_DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT    NOT NULL UNIQUE,
            password_hash TEXT    NOT NULL,
            role          TEXT    NOT NULL DEFAULT 'regular',
            created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    try:
        conn.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (?, ?, 'admin')",
            (username, generate_password_hash(password)),
        )
        conn.commit()
        print(f"Admin user '{username}' created in {USERS_DB_PATH}.")
    except sqlite3.IntegrityError:
        print(f"Error: username '{username}' already exists.")
        sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
