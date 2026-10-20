"""Tools the agent can call. Each tool = a plain Python function + a declaration
(name, description, parameters) that tells the LLM when and how to use it."""
import json
import os
import sqlite3
from contextlib import closing
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

from app.retrieval import search_documents as rag_search

load_dotenv()

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "invoices.db"
MAX_ROWS = 50
STATUSES = ("all", "paid", "unpaid", "overdue")


def today() -> str:
    """Today's date as YYYY-MM-DD. APP_TODAY in .env pins it so answers are reproducible."""
    return os.getenv("APP_TODAY") or date.today().isoformat()


def _query(sql, params=()):
    """Run a SELECT on a READ-ONLY connection and return rows as dicts."""
    uri = f"{DB_PATH.as_uri()}?mode=ro"  # mode=ro: SQLite refuses any write
    with closing(sqlite3.connect(uri, uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(row) for row in conn.execute(sql, params).fetchall()]


def _date_filters(issued_from, issued_to):
    """Build WHERE fragments for an issue-date range. Values go through '?' only."""
    where, params = [], []
    for value, op in ((issued_from, ">="), (issued_to, "<=")):
        if value:
            date.fromisoformat(value)  # raises ValueError unless YYYY-MM-DD
            where.append(f"issue_date {op} ?")
            params.append(value)
    return where, params


# ---------------------------------------------------------------- tools

def list_clients():
    rows = _query("SELECT DISTINCT client FROM invoices ORDER BY client")
    return {"clients": [r["client"] for r in rows]}


def get_invoices(client=None, status="all", issued_from=None, issued_to=None):
    if status not in STATUSES:
        return {"error": f"status must be one of {STATUSES}"}
    where, params = _date_filters(issued_from, issued_to)
    if client:
        where.append("client LIKE ?")
        params.append(f"%{client}%")  # case-insensitive partial match
    if status == "paid":
        where.append("paid = 1")
    elif status in ("unpaid", "overdue"):
        where.append("paid = 0")
    if status == "overdue":
        where.append("due_date < ?")
        params.append(today())

    # Only fixed SQL fragments are joined into the string; every value is a '?' parameter.
    sql = "SELECT id, client, amount_cents, issue_date, due_date, paid, paid_date FROM invoices"
    if where:
        sql += " WHERE " + " AND ".join(where)
    rows = _query(sql + " ORDER BY issue_date", params)

    total_cents = sum(r["amount_cents"] for r in rows)
    for r in rows:
        r["amount_eur"] = r.pop("amount_cents") / 100
        r["paid"] = bool(r["paid"])
    return {
        "today": today(),
        "count": len(rows),
        "total_eur": total_cents / 100,
        "invoices": rows[:MAX_ROWS],
        "truncated": len(rows) > MAX_ROWS,
    }


def client_summary(issued_from=None, issued_to=None):
    where, params = _date_filters(issued_from, issued_to)
    where_sql = (" WHERE " + " AND ".join(where)) if where else ""
    rows = _query(
        f"""SELECT client,
                   COUNT(*) AS invoice_count,
                   SUM(amount_cents) AS billed,
                   SUM(CASE WHEN paid = 0 THEN amount_cents ELSE 0 END) AS unpaid,
                   SUM(CASE WHEN paid = 0 AND due_date < ? THEN amount_cents ELSE 0 END) AS overdue
            FROM invoices{where_sql}
            GROUP BY client
            ORDER BY billed DESC""",
        [today(), *params],
    )
    for r in rows:
        for key in ("billed", "unpaid", "overdue"):
            r[f"{key}_eur"] = r.pop(key) / 100
    return {"today": today(), "issued_from": issued_from, "issued_to": issued_to, "clients": rows}


def search_documents(query):
    if not query or not query.strip():
        return {"error": "query must not be empty"}
    hits = rag_search(query, k=3)
    return {"results": [{"source": h["source"], "section": h["section"], "text": h["text"]} for h in hits]}


# ---------------------------------------------------------------- declarations for the LLM

DATE_PARAM = "Date in YYYY-MM-DD format, inclusive. Omit for no limit."

TOOL_DECLARATIONS = [
    {
        "name": "list_clients",
        "description": "List the exact names of all clients in the invoice database. "
                       "Use when the user mentions a client by a partial or informal name.",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "get_invoices",
        "description": "Look up individual invoices from the user's own invoice database, filtered by "
                       "client, payment status and issue-date range. Returns the matching invoices plus "
                       "their count and total in EUR (already summed - do not re-add). Use for questions "
                       "like 'which invoices are unpaid', 'show my September invoices', 'what is overdue'.",
        "parameters": {
            "type": "object",
            "properties": {
                "client": {"type": "string", "description": "Client name or part of it, case-insensitive. Omit for all clients."},
                "status": {"type": "string", "enum": list(STATUSES),
                           "description": "'overdue' means unpaid and past the due date. Default 'all'."},
                "issued_from": {"type": "string", "description": "Earliest issue date. " + DATE_PARAM},
                "issued_to": {"type": "string", "description": "Latest issue date. " + DATE_PARAM},
            },
        },
    },
    {
        "name": "client_summary",
        "description": "Per-client totals from the user's own invoice database: amount billed, unpaid and "
                       "overdue in EUR, sorted from the biggest client down. Use for 'how much does X owe me', "
                       "'who is my biggest client', and other totals or rankings.",
        "parameters": {
            "type": "object",
            "properties": {
                "issued_from": {"type": "string", "description": "Only count invoices issued on/after this date. " + DATE_PARAM},
                "issued_to": {"type": "string", "description": "Only count invoices issued on/before this date. " + DATE_PARAM},
            },
        },
    },
    {
        "name": "search_documents",
        "description": "Search the tax and invoicing rules documents (VAT rates, payment deadlines, late "
                       "payment, reverse charge, invoice requirements). Use for questions about rules or "
                       "definitions, NOT about the user's own invoices. Returns the most relevant passages, "
                       "which may not contain the answer.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The question, in natural language."}},
            "required": ["query"],
        },
    },
]

TOOLS = {
    "list_clients": list_clients,
    "get_invoices": get_invoices,
    "client_summary": client_summary,
    "search_documents": search_documents,
}


def run_tool(name, args):
    """Run a tool by name. Only tools in TOOLS can run; errors come back as data, not crashes."""
    if name not in TOOLS:
        return {"error": f"unknown tool: {name}"}
    try:
        return TOOLS[name](**(args or {}))
    except (TypeError, ValueError) as e:
        return {"error": f"bad arguments for {name}: {e}"}
    except Exception as e:
        return {"error": f"{name} failed ({type(e).__name__})"}


# ---------------------------------------------------------------- manual test

if __name__ == "__main__":
    demos = [
        ("list_clients", {}),
        ("client_summary", {"issued_from": "2026-01-01"}),
        ("get_invoices", {"client": "alice", "status": "unpaid"}),
        ("get_invoices", {"status": "unpaid", "issued_from": "2026-09-01", "issued_to": "2026-09-30"}),
        ("get_invoices", {"status": "overdue"}),
        ("get_invoices", {"status": "late"}),            # invalid on purpose
        ("get_invoices", {"issued_from": "September"}),  # invalid on purpose
        ("delete_everything", {}),                       # not a tool
        ("search_documents", {"query": "standard VAT rate"}),
    ]
    for name, args in demos:
        result = run_tool(name, args)
        summary = {k: v for k, v in result.items() if k != "invoices"}  # keep output short
        print(f"\n>>> {name}({args})\n{json.dumps(summary, indent=2, ensure_ascii=False)}")

    try:
        _query("DELETE FROM invoices")
        print("\nFAIL: delete succeeded!")
    except sqlite3.OperationalError as e:
        print(f"\nOK: write blocked by read-only connection ({e})")