import os
import sqlite3
import threading
import webbrowser
from datetime import datetime, timezone
from functools import wraps

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, Response, flash, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "mudra_model.keras")
VAL_DIR = os.path.join(BASE_DIR, "..", "val")
DB_PATH = os.path.join(BASE_DIR, "app_data.db")
CLASS_NAMES = ["Alapadmam", "Chandrakala", "Kapitham", "Shikaram"]
IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 60.0
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

APP_USERNAME = os.getenv("MUDRA_APP_USERNAME", "admin")
APP_PASSWORD = os.getenv("MUDRA_APP_PASSWORD", "mudra123")

MUDRA_GUIDE = {
    "Alapadmam": "Blooming lotus hand. Conveys beauty, grace, and unfolding expression in classical dance.",
    "Chandrakala": "Crescent moon hand. Represents moonlight, calm rhythms, and lyrical softness.",
    "Kapitham": "Compact hand form used in devotional and narrative expressions.",
    "Shikaram": "Strong thumb-raised fist. Symbolizes firmness, structure, and determination.",
    "Pataka": "Flat palm gesture used to depict clouds, blessings, or ceremonial movement.",
    "Tripataka": "Three-point variation of pataka, often used for crowns, trees, and flames.",
    "Ardhachandra": "Half-moon shape used for gentle directional and devotional motifs.",
    "Katakamukha": "Delicate, ornament-like hand shape often associated with garlands or adornment.",
}

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "classical-ui-secret")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)
camera = None
camera_lock = threading.Lock()
last_frame = None


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user',
            created_at TEXT NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS login_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            username TEXT NOT NULL,
            logged_in_at TEXT NOT NULL,
            ip_address TEXT,
            user_agent TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """
    )
    conn.commit()
    conn.close()


def ensure_admin_user():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE username = ?", (APP_USERNAME,))
    row = cur.fetchone()
    if row is None:
        cur.execute(
            "INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, ?, ?)",
            (APP_USERNAME, generate_password_hash(APP_PASSWORD), "admin", datetime.now(timezone.utc).isoformat()),
        )
    else:
        cur.execute("UPDATE users SET role = 'admin' WHERE username = ?", (APP_USERNAME,))
    conn.commit()
    conn.close()


def get_user_by_username(username: str):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, username, password_hash, role FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()
    return row


def create_user(username: str, password: str):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, 'user', ?)",
        (username, generate_password_hash(password), datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()


def record_login_event(user_id: int, username: str):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO login_events (user_id, username, logged_in_at, ip_address, user_agent) VALUES (?, ?, ?, ?, ?)",
        (
            user_id,
            username,
            datetime.now(timezone.utc).isoformat(),
            request.remote_addr,
            request.headers.get("User-Agent", "unknown"),
        ),
    )
    conn.commit()
    conn.close()


def get_all_login_events():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT username, logged_in_at, ip_address, user_agent
        FROM login_events
        ORDER BY id DESC
        """
    )
    rows = cur.fetchall()
    conn.close()
    return rows


init_db()
ensure_admin_user()


def login_required(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not session.get("is_authenticated"):
            return redirect(url_for("login"))
        return func(*args, **kwargs)

    return wrapper


def admin_required(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not session.get("is_authenticated"):
            return redirect(url_for("login"))
        if session.get("role") != "admin":
            flash("Admin access only.", "error")
            return redirect(url_for("dashboard"))
        return func(*args, **kwargs)

    return wrapper


def predict_from_rgb(rgb_image: np.ndarray):
    resized = cv2.resize(rgb_image, IMG_SIZE)
    batch = np.expand_dims(resized.astype(np.float32), axis=0)
    preds = model.predict(batch, verbose=0)[0]

    top_idx = int(np.argmax(preds))
    mudra = CLASS_NAMES[top_idx]
    confidence = float(preds[top_idx] * 100.0)
    probabilities = {CLASS_NAMES[i]: float(preds[i] * 100.0) for i in range(len(CLASS_NAMES))}
    return mudra, confidence, probabilities

def has_probable_hand(bgr_image: np.ndarray) -> bool:
    """Heuristic hand presence check to avoid false predictions on empty frames."""
    if bgr_image is None or bgr_image.size == 0:
        return False

    h, w = bgr_image.shape[:2]
    frame_area = float(h * w)

    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2YCrCb)

    
    hsv_mask_low = cv2.inRange(hsv, (0, 20, 45), (25, 255, 255))
    hsv_mask_high = cv2.inRange(hsv, (160, 20, 45), (180, 255, 255))
    hsv_mask = cv2.bitwise_or(hsv_mask_low, hsv_mask_high)
    ycrcb_mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 185, 135))
    skin_mask = cv2.bitwise_and(hsv_mask, ycrcb_mask)

    kernel = np.ones((5, 5), np.uint8)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)

    skin_ratio = cv2.countNonZero(skin_mask) / frame_area
    if skin_ratio < 0.012 or skin_ratio > 0.65:
        return False

    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False

    largest_area = max(cv2.contourArea(c) for c in contours)
    return largest_area > (0.01 * frame_area)


def ensure_camera():
    global camera
    with camera_lock:
        if camera is None or not camera.isOpened():
            camera = cv2.VideoCapture(0)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return camera


def get_live_result():
    return session.get("live_result")


def get_upload_result():
    return session.get("upload_result")


def store_result(source, mudra, confidence, probabilities):
    result = {
        "source": source,
        "mudra": mudra,
        "confidence": round(confidence, 2),
        "is_confident": confidence >= CONFIDENCE_THRESHOLD,
        "probabilities": {name: round(value, 2) for name, value in probabilities.items()},
    }

    if source == "live camera":
        session["live_result"] = result
    elif source == "upload":
        session["upload_result"] = result


def get_val_image_choices():
    choices = []
    if not os.path.isdir(VAL_DIR):
        return choices

    for root, _, files in os.walk(VAL_DIR):
        for filename in files:
            ext = os.path.splitext(filename)[1].lower()
            if ext in ALLOWED_EXTENSIONS:
                full_path = os.path.join(root, filename)
                rel_path = os.path.relpath(full_path, VAL_DIR).replace("\\", "/")
                choices.append(rel_path)

    choices.sort()
    return choices


def mjpeg_stream():
    global last_frame
    video = ensure_camera()

    while True:
        success, frame = video.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)

        with camera_lock:
            last_frame = frame.copy()

        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )


@app.route("/")
def home():
    return redirect(url_for("intro")) if session.get("is_authenticated") else redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        user = get_user_by_username(username)
        if user is not None and check_password_hash(user["password_hash"], password):
            session["is_authenticated"] = True
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            session["role"] = user["role"]
            record_login_event(user["id"], user["username"])
            return redirect(url_for("intro"))

        flash("Invalid username or password.", "error")

    return render_template("login.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if len(username) < 3:
            flash("Username must be at least 3 characters.", "error")
            return render_template("signup.html")

        if len(password) < 6:
            flash("Password must be at least 6 characters.", "error")
            return render_template("signup.html")

        if password != confirm_password:
            flash("Passwords do not match.", "error")
            return render_template("signup.html")

        if get_user_by_username(username) is not None:
            flash("Username already exists. Please choose another.", "error")
            return render_template("signup.html")

        create_user(username, password)
        flash("Account created successfully. Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("signup.html")


@app.route("/intro")
@login_required
def intro():
    return render_template(
        "intro.html",
        username=session.get("username", "Dancer"),
        is_admin=session.get("role") == "admin",
        mudra_guide=MUDRA_GUIDE,
    )


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/dashboard")
@login_required
def dashboard():
    return render_template(
        "dashboard.html",
        username=session.get("username", "Dancer"),
        is_admin=session.get("role") == "admin",
        class_names=CLASS_NAMES,
        threshold=CONFIDENCE_THRESHOLD,
        mudra_guide=MUDRA_GUIDE,
        live_result=get_live_result(),
        upload_result=get_upload_result(),
        val_dir_display=os.path.abspath(VAL_DIR),
        val_images=get_val_image_choices(),
    )


@app.route("/video-feed")
@login_required
def video_feed():
    return Response(mjpeg_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/detect-live", methods=["POST"])
@login_required
def detect_live():
    global last_frame

    with camera_lock:
        current_frame = None if last_frame is None else last_frame.copy()

    if current_frame is None:
        flash("Camera frame is not ready yet. Open the dashboard and wait a moment.", "error")
        return redirect(url_for("dashboard"))

    if not has_probable_hand(current_frame):
        flash("No clear hand gesture detected in frame. Please show your hand and try again.", "error")
        return redirect(url_for("dashboard"))

    img_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
    mudra, confidence, probabilities = predict_from_rgb(img_rgb)
    store_result("live camera", mudra, confidence, probabilities)
    return redirect(url_for("dashboard"))


@app.route("/predict-upload", methods=["POST"])
@login_required
def predict_upload():
    selected_image = request.form.get("selected_image", "").strip()
    
    if not selected_image:
        flash("Please select an image from the val folder.", "error")
        return redirect(url_for("dashboard"))
    
    
    full_path = os.path.normpath(os.path.join(VAL_DIR, selected_image))
    val_dir_normalized = os.path.normpath(os.path.abspath(VAL_DIR))
    full_path_normalized = os.path.normpath(os.path.abspath(full_path))
    
    # Ensure the resolved path is within VAL_DIR
    if not full_path_normalized.startswith(val_dir_normalized):
        flash("Invalid image path. Must be from val folder.", "error")
        return redirect(url_for("dashboard"))
    
    if not os.path.exists(full_path_normalized):
        flash("Selected image file not found.", "error")
        return redirect(url_for("dashboard"))
    
    ext = os.path.splitext(full_path_normalized)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        flash("Only JPG and PNG images are supported.", "error")
        return redirect(url_for("dashboard"))
    
    # Read and predict
    img_bgr = cv2.imread(full_path_normalized)
    if img_bgr is None:
        flash("Could not read selected image.", "error")
        return redirect(url_for("dashboard"))

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mudra, confidence, probabilities = predict_from_rgb(img_rgb)
    store_result("upload", mudra, confidence, probabilities)
    return redirect(url_for("dashboard"))


@app.route("/health")
def health():
    return {"ok": True, "model": os.path.basename(MODEL_PATH), "classes": CLASS_NAMES}


@app.route("/admin/logins")
@admin_required
def admin_logins():
    return render_template(
        "admin_logins.html",
        username=session.get("username", "admin"),
        login_events=get_all_login_events(),
    )


if __name__ == "__main__":
    
    threading.Timer(1.0, lambda: webbrowser.open("http://127.0.0.1:5000")).start()
    app.run(debug=True)
