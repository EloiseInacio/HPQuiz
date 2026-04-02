import os
import sqlite3
import string

from flask import Flask, redirect, render_template, request, session, url_for

APP_SECRET_KEY = os.environ.get("FLASK_SECRET", "hp-quiz-dev-secret-change-me")
DB_PATH        = os.environ.get("HPQUIZ_DB", "questions.db")
DIFFICULTIES   = ("easy", "medium", "hard", "any")
MAX_QUESTIONS  = 20
MIN_QUESTIONS  = 1

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


def get_questions(difficulty: str, count: int) -> list:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cols = "id, question, answer, difficulty, COALESCE(book,'') AS book, COALESCE(chapter,'') AS chapter"
    if difficulty == "any":
        rows = conn.execute(
            f"SELECT {cols} FROM questions ORDER BY RANDOM() LIMIT ?",
            (count,),
        ).fetchall()
    else:
        rows = conn.execute(
            f"SELECT {cols} FROM questions WHERE difficulty = ? ORDER BY RANDOM() LIMIT ?",
            (difficulty, count),
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def score_answer(user_answer: str, correct_answer: str) -> bool:
    table = str.maketrans("", "", string.punctuation)
    user_tokens    = set(user_answer.lower().translate(table).split()) - STOPWORDS
    correct_tokens = set(correct_answer.lower().translate(table).split()) - STOPWORDS
    if not correct_tokens or not user_tokens:
        return correct_answer.lower() in user_answer.lower()
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
    for key in ("quiz_questions", "quiz_index", "quiz_results", "quiz_total"):
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

    questions = get_questions(difficulty, count)
    if not questions:
        session["flash_message"] = "No questions found for that difficulty! 😢 Try a different one or run generate_questions.py first."
        return redirect(url_for("index"))

    session["quiz_questions"] = questions
    session["quiz_index"]     = 0
    session["quiz_total"]     = len(questions)
    session["quiz_results"]   = []
    return redirect(url_for("question"))


@app.route("/question")
def question():
    if "quiz_questions" not in session:
        return redirect(url_for("index"))
    if session["quiz_index"] >= session["quiz_total"]:
        return redirect(url_for("summary"))
    idx = session["quiz_index"]
    q   = session["quiz_questions"][idx]
    return render_template("question.html", question=q,
                           index=idx + 1, total=session["quiz_total"],
                           show_result=False)


@app.route("/submit", methods=["POST"])
def submit():
    if "quiz_questions" not in session:
        return redirect(url_for("index"))
    idx = session["quiz_index"]
    if idx >= session["quiz_total"]:
        return redirect(url_for("summary"))

    # guard against double-submission on browser Back
    if len(session["quiz_results"]) > idx:
        result = session["quiz_results"][idx]
        q = session["quiz_questions"][idx]
        return render_template("question.html", question=q,
                               index=idx + 1, total=session["quiz_total"],
                               show_result=True,
                               user_answer=result["user_answer"],
                               is_correct=result["is_correct"],
                               correct_answer=result["correct_answer"])

    q           = session["quiz_questions"][idx]
    user_answer = request.form.get("user_answer", "").strip()
    is_correct  = score_answer(user_answer, q["answer"])

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
    if "quiz_questions" not in session:
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
    for key in ("quiz_questions", "quiz_index", "quiz_results", "quiz_total"):
        session.pop(key, None)
    return render_template("summary.html", results=results,
                           correct_count=correct_count, total=total,
                           pct=pct, message=message)
