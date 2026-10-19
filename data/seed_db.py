"""Create data/invoices.db with sample invoices. Safe to re-run: rebuilds from scratch."""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "invoices.db"

SCHEMA = """
CREATE TABLE invoices (
    id           INTEGER PRIMARY KEY,
    client       TEXT    NOT NULL,
    amount_cents INTEGER NOT NULL CHECK (amount_cents > 0),
    issue_date   TEXT    NOT NULL,          -- ISO: YYYY-MM-DD
    due_date     TEXT    NOT NULL,
    paid         INTEGER NOT NULL CHECK (paid IN (0, 1)),
    paid_date    TEXT,                      -- NULL while unpaid
    CHECK (due_date >= issue_date),
    CHECK ((paid = 1) = (paid_date IS NOT NULL))
);
"""

# (client, amount_cents, issue_date, due_date, paid_date or None if unpaid)
INVOICES = [
    ("Alice's Accounting",   480000, "2025-11-12", "2025-12-12", "2025-12-10"),
    ("Alice's Accounting",   520000, "2026-01-15", "2026-02-14", "2026-02-20"),
    ("Alice's Accounting",   610000, "2026-04-03", "2026-05-03", "2026-05-02"),
    ("Alice's Accounting",   455000, "2026-07-08", "2026-08-07", "2026-08-30"),
    ("Alice's Accounting",   390000, "2026-09-10", "2026-10-10", None),
    ("Alice's Accounting",   275000, "2026-10-05", "2026-11-04", None),
    ("Bob's Bakery",          45000, "2026-02-02", "2026-02-17", "2026-02-16"),
    ("Bob's Bakery",          52000, "2026-05-04", "2026-05-19", "2026-05-19"),
    ("Bob's Bakery",          48000, "2026-08-03", "2026-08-18", "2026-08-25"),
    ("Bob's Bakery",          51000, "2026-09-01", "2026-09-16", None),
    ("Carol's Cafe",         120000, "2025-12-01", "2025-12-31", "2025-12-28"),
    ("Carol's Cafe",         135000, "2026-03-09", "2026-04-08", "2026-04-08"),
    ("Carol's Cafe",          98000, "2026-06-15", "2026-07-15", "2026-07-10"),
    ("Carol's Cafe",         142000, "2026-09-14", "2026-10-14", None),
    ("Carol's Cafe",          87000, "2026-09-28", "2026-10-28", None),
    ("Dan's Diner",          230000, "2026-01-20", "2026-02-19", "2026-03-15"),
    ("Dan's Diner",          180000, "2026-04-22", "2026-05-22", "2026-06-10"),
    ("Dan's Diner",          210000, "2026-07-01", "2026-07-31", "2026-08-20"),
    ("Dan's Diner",          195000, "2026-09-22", "2026-10-22", None),
    ("Earl's Engineering",   350000, "2026-03-17", "2026-04-16", "2026-04-14"),
    ("Earl's Engineering",   420000, "2026-06-02", "2026-07-02", "2026-07-01"),
    ("Earl's Engineering",   380000, "2026-09-05", "2026-10-05", "2026-10-01"),
    ("Earl's Engineering",   300000, "2026-10-12", "2026-11-11", None),
]

SUMMARY_SQL = """
SELECT client,
       COUNT(*)                                           AS n,
       SUM(amount_cents) / 100.0                          AS total_eur,
       SUM(CASE WHEN paid = 0 THEN amount_cents END) / 100.0 AS unpaid_eur
FROM invoices
WHERE issue_date >= '2026-01-01'
GROUP BY client
ORDER BY total_eur DESC;
"""


def main():
    DB_PATH.unlink(missing_ok=True)  # start fresh every run
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    conn.executemany(
        "INSERT INTO invoices (client, amount_cents, issue_date, due_date, paid, paid_date) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [(c, a, i, d, 1 if p else 0, p) for c, a, i, d, p in INVOICES],
    )
    conn.commit()

    total = conn.execute("SELECT COUNT(*) FROM invoices").fetchone()[0]
    print(f"Created {DB_PATH.name} with {total} invoices.\n\n2026 summary:")
    for client, n, total_eur, unpaid_eur in conn.execute(SUMMARY_SQL):
        print(f"  {client:<22} {n} invoices  {total_eur:>10,.2f} EUR   unpaid {unpaid_eur or 0:>9,.2f} EUR")
    conn.close()


if __name__ == "__main__":
    main()