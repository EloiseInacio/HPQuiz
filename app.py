import os
import re
import sqlite3
import string
import subprocess
import time
from functools import wraps

from flask import Flask, jsonify, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

APP_SECRET_KEY  = os.environ.get("FLASK_SECRET", "hp-quiz-dev-secret-change-me")
DB_PATH         = os.environ.get("HPQUIZ_DB", "questions_semantic.db")
USERS_DB_PATH   = os.environ.get("HPQUIZ_USERS_DB", "users.db")
GENERATION_LOG  = "generation.log"
DIFFICULTIES    = ("easy", "medium", "hard", "any")
MAX_QUESTIONS   = 20
MIN_QUESTIONS   = 1
ADMIN_PAGE_SIZE  = 20
DIFFICULTY_STEP  = 2   # similarity_count delta per answer
DIFFICULTY_THRESHOLDS = {"easy": 15, "medium": 5}   # mirrors generate_questions.py

_generation_proc = None

STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by",
    "did", "do", "for", "from", "had", "has", "have", "he", "her",
    "him", "his", "how", "i", "in", "is", "it", "its", "of", "on",
    "or", "our", "she", "so", "that", "the", "their", "them", "they",
    "this", "to", "was", "we", "were", "what", "when", "where", "who",
    "why", "with", "you", "your",
})

app = Flask(__name__)
app.secret_key = APP_SECRET_KEY


def get_users_db():
    conn = sqlite3.connect(USERS_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT    NOT NULL UNIQUE,
            password_hash TEXT    NOT NULL,
            role          TEXT    NOT NULL DEFAULT 'regular',
            created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS quiz_results (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            score_pct  INTEGER NOT NULL,
            correct    INTEGER NOT NULL,
            total      INTEGER NOT NULL,
            duration_s INTEGER NOT NULL,
            difficulty TEXT    NOT NULL,
            played_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_quiz_results_user ON quiz_results(user_id)"
    )
    conn.commit()
    return conn


def save_quiz_result(user_id: int, score_pct: int, correct: int, total: int,
                     duration_s: int, difficulty: str) -> None:
    conn = get_users_db()
    conn.execute(
        "INSERT INTO quiz_results (user_id, score_pct, correct, total, duration_s, difficulty)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (user_id, score_pct, correct, total, duration_s, difficulty),
    )
    conn.commit()
    conn.close()


def get_user_stats(user_id: int) -> tuple:
    conn = get_users_db()
    history = conn.execute(
        "SELECT score_pct, correct, total, duration_s, difficulty, played_at"
        " FROM quiz_results WHERE user_id = ? ORDER BY played_at DESC LIMIT 20",
        (user_id,),
    ).fetchall()
    agg = conn.execute(
        "SELECT COUNT(*), MAX(score_pct), ROUND(AVG(score_pct)),"
        "       MIN(duration_s), ROUND(AVG(duration_s))"
        " FROM quiz_results WHERE user_id = ?",
        (user_id,),
    ).fetchone()
    conn.close()
    return history, agg


def get_user_by_username(username: str) -> dict | None:
    conn = get_users_db()
    row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()
    return dict(row) if row else None


def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if session.get("role") != "admin":
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


def get_question_ids(difficulty: str, count: int) -> list:
    conn = sqlite3.connect(DB_PATH)
    if difficulty == "any":
        rows = conn.execute(
            "SELECT id FROM questions ORDER BY RANDOM() LIMIT ?", (count,)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id FROM questions WHERE difficulty = ? ORDER BY RANDOM() LIMIT ?",
            (difficulty, count),
        ).fetchall()
    conn.close()
    return [r[0] for r in rows]


def get_question_by_id(qid: int, include_answer: bool = False) -> dict | None:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cols = "id, question, difficulty, COALESCE(book,'') AS book, COALESCE(chapter,'') AS chapter"
    if include_answer:
        cols += ", answer"
    row = conn.execute(f"SELECT {cols} FROM questions WHERE id = ?", (qid,)).fetchone()
    conn.close()
    return dict(row) if row else None


def update_question_difficulty(qid: int, is_correct: bool) -> None:
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT similarity_count FROM questions WHERE id = ?", (qid,)).fetchone()
    if row is None:
        conn.close()
        return
    score = max(0, min(100, row[0] + (DIFFICULTY_STEP if is_correct else -DIFFICULTY_STEP)))
    if score > DIFFICULTY_THRESHOLDS["easy"]:
        difficulty = "easy"
    elif score >= DIFFICULTY_THRESHOLDS["medium"]:
        difficulty = "medium"
    else:
        difficulty = "hard"
    conn.execute(
        "UPDATE questions SET similarity_count = ?, difficulty = ? WHERE id = ?",
        (score, difficulty, qid),
    )
    conn.commit()
    conn.close()


_NUM_ONES = [
    "zero","one","two","three","four","five","six","seven","eight","nine",
    "ten","eleven","twelve","thirteen","fourteen","fifteen","sixteen",
    "seventeen","eighteen","nineteen",
]
_NUM_TENS = ["","","twenty","thirty","forty","fifty","sixty","seventy","eighty","ninety"]


def _int_to_words(n: int) -> str:
    if n < 20:
        return _NUM_ONES[n]
    if n < 100:
        rest = (" " + _NUM_ONES[n % 10]) if n % 10 else ""
        return _NUM_TENS[n // 10] + rest
    if n < 1000:
        rest = (" " + _int_to_words(n % 100)) if n % 100 else ""
        return _NUM_ONES[n // 100] + " hundred" + rest
    return str(n)


def _normalize_numbers(text: str) -> str:
    """Replace digit sequences with their word equivalents (0-999)."""
    return re.sub(
        r"\b(\d+)\b",
        lambda m: _int_to_words(int(m.group())) if int(m.group()) < 1000 else m.group(),
        text,
    )


def score_answer(user_answer: str, correct_answer: str) -> bool:
    table = str.maketrans("", "", string.punctuation)
    user_norm    = _normalize_numbers(user_answer.lower()).translate(table)
    correct_norm = _normalize_numbers(correct_answer.lower()).translate(table)
    user_tokens    = set(user_norm.split()) - STOPWORDS
    correct_tokens = set(correct_norm.split()) - STOPWORDS
    if not correct_tokens or not user_tokens:
        return correct_norm in user_norm
    return len(user_tokens & correct_tokens) >= max(1, len(correct_tokens) // 2)


def get_score_message(pct: int) -> str:
    if pct == 100:
        return "Perfect score! You must be Hermione! 🏆✨"
    if pct >= 80:
        return "Excellent! Dumbledore would be proud! 🧙‍♂️⭐"
    if pct >= 60:
        return "Not bad! You've earned your Hogwarts letter! 🦉📜"
    if pct >= 40:
        return "Room to grow! Time to visit the library! 📚🔍"
    if pct >= 1:
        return "Even Neville started somewhere! Keep practising! 🌱💪"
    return "Did you fall into the Devil's Snare? Try again! 🌿😅"


@app.route("/")
def index():
    for key in ("quiz_question_ids", "quiz_index", "quiz_results", "quiz_total"):
        session.pop(key, None)
    flash_message = session.pop("flash_message", None)
    return render_template("index.html", difficulties=DIFFICULTIES,
                           max_q=MAX_QUESTIONS, min_q=MIN_QUESTIONS,
                           flash_message=flash_message)


@app.route("/start", methods=["POST"])
def start():
    try:
        count = int(request.form.get("count", 10))
    except ValueError:
        count = 10
    count = max(MIN_QUESTIONS, min(MAX_QUESTIONS, count))

    difficulty = request.form.get("difficulty", "any")
    if difficulty not in DIFFICULTIES:
        difficulty = "any"

    question_ids = get_question_ids(difficulty, count)
    if not question_ids:
        session["flash_message"] = "No questions found for that difficulty! 😢 Try a different one or run generate_questions.py first."
        return redirect(url_for("index"))

    session["quiz_question_ids"] = question_ids
    session["quiz_index"]        = 0
    session["quiz_total"]        = len(question_ids)
    session["quiz_results"]      = []
    session["quiz_start"]        = int(time.time())
    session["quiz_difficulty"]   = difficulty
    return redirect(url_for("question"))


@app.route("/question")
def question():
    if "quiz_question_ids" not in session:
        return redirect(url_for("index"))
    if session["quiz_index"] >= session["quiz_total"]:
        return redirect(url_for("summary"))
    idx = session["quiz_index"]
    q   = get_question_by_id(session["quiz_question_ids"][idx])
    if q is None:
        return redirect(url_for("index"))
    return render_template("question.html", question=q,
                           index=idx + 1, total=session["quiz_total"],
                           show_result=False)


@app.route("/submit", methods=["POST"])
def submit():
    if "quiz_question_ids" not in session:
        return redirect(url_for("index"))
    idx = session["quiz_index"]
    if idx >= session["quiz_total"]:
        return redirect(url_for("summary"))

    q = get_question_by_id(session["quiz_question_ids"][idx], include_answer=True)
    if q is None:
        return redirect(url_for("index"))

    # guard against double-submission on browser Back
    if len(session["quiz_results"]) > idx:
        result = session["quiz_results"][idx]
        return render_template("question.html", question=q,
                               index=idx + 1, total=session["quiz_total"],
                               show_result=True,
                               user_answer=result["user_answer"],
                               is_correct=result["is_correct"],
                               correct_answer=result["correct_answer"])

    user_answer = request.form.get("user_answer", "").strip()
    is_correct  = score_answer(user_answer, q["answer"])
    update_question_difficulty(q["id"], is_correct)

    session["quiz_results"].append({
        "question":       q["question"],
        "correct_answer": q["answer"],
        "user_answer":    user_answer,
        "is_correct":     is_correct,
        "difficulty":     q["difficulty"],
    })
    session.modified = True

    return render_template("question.html", question=q,
                           index=idx + 1, total=session["quiz_total"],
                           show_result=True,
                           user_answer=user_answer,
                           is_correct=is_correct,
                           correct_answer=q["answer"])


@app.route("/next", methods=["POST"])
def next_question():
    if "quiz_question_ids" not in session:
        return redirect(url_for("index"))
    session["quiz_index"] += 1
    session.modified = True
    if session["quiz_index"] >= session["quiz_total"]:
        return redirect(url_for("summary"))
    return redirect(url_for("question"))


@app.route("/summary")
def summary():
    if "quiz_results" not in session or not session["quiz_results"]:
        return redirect(url_for("index"))
    results       = session["quiz_results"]
    correct_count = sum(1 for r in results if r["is_correct"])
    total         = len(results)
    pct           = round(correct_count / total * 100)
    message       = get_score_message(pct)
    if session.get("user_id"):
        duration_s = int(time.time()) - session.get("quiz_start", int(time.time()))
        save_quiz_result(session["user_id"], pct, correct_count, total,
                         duration_s, session.get("quiz_difficulty", "any"))
    for key in ("quiz_question_ids", "quiz_index", "quiz_results", "quiz_total",
                "quiz_start", "quiz_difficulty"):
        session.pop(key, None)
    return render_template("summary.html", results=results,
                           correct_count=correct_count, total=total,
                           pct=pct, message=message)


@app.route("/stats")
def stats():
    if not session.get("user_id"):
        return redirect(url_for("login"))
    history, agg = get_user_stats(session["user_id"])
    return render_template("stats.html", history=history, agg=agg)


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template("login.html", error=None)
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")
    user = get_user_by_username(username)
    if user is None or not check_password_hash(user["password_hash"], password):
        return render_template("login.html", error="Invalid username or password.")
    session["user_id"]  = user["id"]
    session["username"] = user["username"]
    session["role"]     = user["role"]
    return redirect(url_for("admin_dashboard") if user["role"] == "admin" else url_for("index"))


@app.route("/logout")
def logout():
    for key in ("user_id", "username", "role"):
        session.pop(key, None)
    return redirect(url_for("index"))


# ---------------------------------------------------------------------------
# Admin panel
# ---------------------------------------------------------------------------

@app.route("/admin")
@admin_required
def admin_dashboard():
    qconn = sqlite3.connect(DB_PATH)
    diff_rows = qconn.execute(
        "SELECT difficulty, COUNT(*) AS n FROM questions GROUP BY difficulty"
    ).fetchall()
    total_q = qconn.execute("SELECT COUNT(*) FROM questions").fetchone()[0]
    qconn.close()
    uconn = get_users_db()
    role_rows = uconn.execute(
        "SELECT role, COUNT(*) AS n FROM users GROUP BY role"
    ).fetchall()
    total_u = uconn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    uconn.close()
    return render_template("admin/dashboard.html",
                           diff_stats=diff_rows, total_q=total_q,
                           role_stats=role_rows, total_u=total_u)


@app.route("/admin/questions")
@admin_required
def admin_questions():
    page = max(1, request.args.get("page", 1, type=int))
    offset = (page - 1) * ADMIN_PAGE_SIZE
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, question, answer, difficulty, COALESCE(book,'') AS book FROM questions"
        " ORDER BY id DESC LIMIT ? OFFSET ?",
        (ADMIN_PAGE_SIZE, offset),
    ).fetchall()
    total = conn.execute("SELECT COUNT(*) FROM questions").fetchone()[0]
    conn.close()
    pages = (total + ADMIN_PAGE_SIZE - 1) // ADMIN_PAGE_SIZE
    return render_template("admin/questions.html",
                           questions=rows, page=page, pages=pages, total=total)


@app.route("/admin/questions/delete/<int:qid>", methods=["POST"])
@admin_required
def admin_delete_question(qid):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM questions WHERE id = ?", (qid,))
    conn.commit()
    conn.close()
    return redirect(request.referrer or url_for("admin_questions"))


@app.route("/admin/users")
@admin_required
def admin_users():
    conn = get_users_db()
    users = conn.execute("SELECT id, username, role, created_at FROM users ORDER BY id").fetchall()
    conn.close()
    return render_template("admin/users.html", users=users)


@app.route("/admin/users/create", methods=["POST"])
@admin_required
def admin_create_user():
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")
    role     = request.form.get("role", "regular")
    if role not in ("admin", "regular"):
        role = "regular"
    if not username or not password:
        return redirect(url_for("admin_users"))
    conn = get_users_db()
    try:
        conn.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
            (username, generate_password_hash(password), role),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        pass  # duplicate username — silently ignore
    finally:
        conn.close()
    return redirect(url_for("admin_users"))


@app.route("/admin/users/delete/<int:uid>", methods=["POST"])
@admin_required
def admin_delete_user(uid):
    if uid == session.get("user_id"):
        return redirect(url_for("admin_users"))  # cannot delete self
    conn = get_users_db()
    conn.execute("DELETE FROM users WHERE id = ?", (uid,))
    conn.commit()
    conn.close()
    return redirect(url_for("admin_users"))


@app.route("/admin/users/role/<int:uid>", methods=["POST"])
@admin_required
def admin_toggle_role(uid):
    if uid == session.get("user_id"):
        return redirect(url_for("admin_users"))  # cannot demote self
    conn = get_users_db()
    row = conn.execute("SELECT role FROM users WHERE id = ?", (uid,)).fetchone()
    if row:
        new_role = "regular" if row["role"] == "admin" else "admin"
        conn.execute("UPDATE users SET role = ? WHERE id = ?", (new_role, uid))
        conn.commit()
    conn.close()
    return redirect(url_for("admin_users"))


@app.route("/admin/generate")
@admin_required
def admin_generate():
    running = _generation_proc is not None and _generation_proc.poll() is None
    log = ""
    if os.path.exists(GENERATION_LOG):
        with open(GENERATION_LOG) as f:
            lines = f.readlines()
        log = "".join(lines[-50:])
    return render_template("admin/generate.html", running=running, log=log,
                           default_collection="hp_semantic",
                           default_output=DB_PATH)


@app.route("/admin/generate/run", methods=["POST"])
@admin_required
def admin_generate_run():
    global _generation_proc
    if _generation_proc is not None and _generation_proc.poll() is None:
        return redirect(url_for("admin_generate"))  # already running
    n          = request.form.get("n", "50")
    collection = request.form.get("collection", "hp_semantic")
    output     = request.form.get("output", DB_PATH)
    # validate numeric n
    try:
        n = str(max(1, int(n)))
    except ValueError:
        n = "50"
    with open(GENERATION_LOG, "w") as log:
        _generation_proc = subprocess.Popen(
            ["conda", "run", "-n", "hpquiz", "python", "generate_questions.py",
             "--n", n, "--collection", collection, "--output", output],
            stdout=log, stderr=log,
        )
    return redirect(url_for("admin_generate"))


@app.route("/admin/generate/status")
@admin_required
def admin_generate_status():
    running = _generation_proc is not None and _generation_proc.poll() is None
    log = ""
    if os.path.exists(GENERATION_LOG):
        with open(GENERATION_LOG) as f:
            lines = f.readlines()
        log = "".join(lines[-50:])
    return jsonify({"running": running, "log": log})
