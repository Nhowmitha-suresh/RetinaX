import os
import sqlite3
import datetime
from typing import Optional, List, Dict
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER", "dr_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "dr_password_2024")
DB_NAME = os.getenv("DB_NAME", "BLINDNESS")
SQLITE_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "retinax.db")

def get_db_connection():
    """
    Attempt MySQL connection first.
    If MySQL is unreachable, fall back to SQLite (retinax.db) for 100% offline persistence.
    """
    try:
        import mysql.connector
        conn = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            connection_timeout=2
        )
        return conn, "mysql"
    except Exception:
        # Fall back to SQLite connection
        conn = sqlite3.connect(os.path.abspath(SQLITE_DB_PATH))
        conn.row_factory = sqlite3.Row
        return conn, "sqlite"

def init_db():
    """Ensure database, patient, prediction, report, and audit tables exist."""
    conn, db_type = get_db_connection()
    if conn is None:
        print("[!] Database unreachable. Skipping DB init.")
        return False
    try:
        cursor = conn.cursor()
        
        if db_type == "mysql":
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{DB_NAME}`")
            cursor.execute(f"USE `{DB_NAME}`")
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS `patients` (
                    `id` INT AUTO_INCREMENT PRIMARY KEY,
                    `patient_id` VARCHAR(100) UNIQUE,
                    `patient_name` VARCHAR(255),
                    `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS `predictions` (
                    `id` INT AUTO_INCREMENT PRIMARY KEY,
                    `patient_name` VARCHAR(255),
                    `patient_id` VARCHAR(100),
                    `prediction_level` INT,
                    `prediction_label` VARCHAR(100),
                    `confidence` FLOAT,
                    `risk_level` VARCHAR(50),
                    `quality_score` FLOAT DEFAULT 85.0,
                    `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS `reports` (
                    `id` INT AUTO_INCREMENT PRIMARY KEY,
                    `report_id` VARCHAR(100) UNIQUE,
                    `patient_id` VARCHAR(100),
                    `prediction_level` INT,
                    `prediction_label` VARCHAR(100),
                    `confidence` FLOAT,
                    `report_filename` VARCHAR(255),
                    `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS `audit_logs` (
                    `id` INT AUTO_INCREMENT PRIMARY KEY,
                    `action` VARCHAR(255),
                    `patient_id` VARCHAR(100),
                    `details` TEXT,
                    `timestamp` TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
        else:
            # SQLite Table Initialization
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS patients (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id TEXT UNIQUE,
                    patient_name TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_name TEXT,
                    patient_id TEXT,
                    prediction_level INTEGER,
                    prediction_label TEXT,
                    confidence REAL,
                    risk_level TEXT,
                    quality_score REAL DEFAULT 85.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    report_id TEXT UNIQUE,
                    patient_id TEXT,
                    prediction_level INTEGER,
                    prediction_label TEXT,
                    confidence REAL,
                    report_filename TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    action TEXT,
                    patient_id TEXT,
                    details TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
        conn.commit()
        cursor.close()
        conn.close()
        print(f"[+] Database initialized successfully ({db_type.upper()}).")
        return True
    except Exception as e:
        print(f"[!] Database initialization failed: {e}")
        return False

def log_prediction(patient_name: str, patient_id: str, level: int, label: str, confidence: float, risk: str, quality_score: float = 85.0):
    """Log prediction record to DB."""
    conn, db_type = get_db_connection()
    if conn is None:
        return False
    try:
        cursor = conn.cursor()
        # Ensure patient exists in patients table
        if db_type == "mysql":
            cursor.execute("INSERT IGNORE INTO patients (patient_id, patient_name) VALUES (%s, %s)", (patient_id, patient_name))
            cursor.execute("""
                INSERT INTO predictions (patient_name, patient_id, prediction_level, prediction_label, confidence, risk_level, quality_score)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (patient_name, patient_id, level, label, confidence, risk, quality_score))
        else:
            cursor.execute("INSERT OR IGNORE INTO patients (patient_id, patient_name) VALUES (?, ?)", (patient_id, patient_name))
            cursor.execute("""
                INSERT INTO predictions (patient_name, patient_id, prediction_level, prediction_label, confidence, risk_level, quality_score)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (patient_name, patient_id, level, label, confidence, risk, quality_score))
            
        conn.commit()
        cursor.close()
        conn.close()
        log_audit_action("PREDICTION_SAVED", patient_id, f"Predicted {label} ({confidence:.1f}%)")
        return True
    except Exception as e:
        print(f"[!] Failed to log prediction: {e}")
        return False

def log_report(report_id: str, patient_id: str, level: int, label: str, confidence: float, filename: str):
    """Log clinical report generation to DB."""
    conn, db_type = get_db_connection()
    if conn is None:
        return False
    try:
        cursor = conn.cursor()
        if db_type == "mysql":
            cursor.execute("""
                INSERT INTO reports (report_id, patient_id, prediction_level, prediction_label, confidence, report_filename)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (report_id, patient_id, level, label, confidence, filename))
        else:
            cursor.execute("""
                INSERT INTO reports (report_id, patient_id, prediction_level, prediction_label, confidence, report_filename)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (report_id, patient_id, level, label, confidence, filename))
        conn.commit()
        cursor.close()
        conn.close()
        log_audit_action("REPORT_GENERATED", patient_id, f"Generated Report ID {report_id}")
        return True
    except Exception as e:
        print(f"[!] Failed to log report: {e}")
        return False

def log_audit_action(action: str, patient_id: str, details: str):
    """Save record into audit_logs table."""
    try:
        conn, db_type = get_db_connection()
        cursor = conn.cursor()
        if db_type == "mysql":
            cursor.execute("INSERT INTO audit_logs (action, patient_id, details) VALUES (%s, %s, %s)", (action, patient_id, details))
        else:
            cursor.execute("INSERT INTO audit_logs (action, patient_id, details) VALUES (?, ?, ?)", (action, patient_id, details))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception:
        pass

def get_report_by_id(report_id: str) -> Optional[Dict]:
    """Fetch report details for QR verification."""
    conn, db_type = get_db_connection()
    if conn is None:
        return None
    try:
        cursor = conn.cursor()
        if db_type == "mysql":
            cursor.execute("SELECT * FROM reports WHERE report_id = %s", (report_id,))
            row = cursor.fetchone()
            if row:
                return {
                    "report_id": row[1],
                    "patient_id": row[2],
                    "prediction_level": row[3],
                    "prediction_label": row[4],
                    "confidence": row[5],
                    "report_filename": row[6],
                    "created_at": str(row[7])
                }
        else:
            cursor.execute("SELECT * FROM reports WHERE report_id = ?", (report_id,))
            row = cursor.fetchone()
            if row:
                return {
                    "report_id": row["report_id"],
                    "patient_id": row["patient_id"],
                    "prediction_level": row["prediction_level"],
                    "prediction_label": row["prediction_label"],
                    "confidence": row["confidence"],
                    "report_filename": row["report_filename"],
                    "created_at": str(row["created_at"])
                }
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"[!] Error fetching report {report_id}: {e}")
    return None

def get_patient_history(patient_id: str = None) -> List[Dict]:
    """Fetch screening records for history & timeline."""
    conn, db_type = get_db_connection()
    if conn is None:
        return []
    records = []
    try:
        cursor = conn.cursor()
        if patient_id:
            query = "SELECT * FROM predictions WHERE patient_id = ? ORDER BY id DESC" if db_type == "sqlite" else "SELECT * FROM predictions WHERE patient_id = %s ORDER BY id DESC"
            cursor.execute(query, (patient_id,))
        else:
            cursor.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT 50")
            
        rows = cursor.fetchall()
        for r in rows:
            if db_type == "sqlite":
                records.append({
                    "id": r["id"],
                    "patient_name": r["patient_name"],
                    "patient_id": r["patient_id"],
                    "prediction_level": r["prediction_level"],
                    "prediction_label": r["prediction_label"],
                    "confidence": r["confidence"],
                    "risk_level": r["risk_level"],
                    "quality_score": r["quality_score"],
                    "created_at": str(r["created_at"])
                })
            else:
                records.append({
                    "id": r[0],
                    "patient_name": r[1],
                    "patient_id": r[2],
                    "prediction_level": r[3],
                    "prediction_label": r[4],
                    "confidence": r[5],
                    "risk_level": r[6],
                    "quality_score": r[7],
                    "created_at": str(r[8])
                })
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"[!] Error fetching patient history: {e}")
    return records

def get_dashboard_statistics() -> Dict:
    """Compile doctor dashboard screening statistics."""
    conn, db_type = get_db_connection()
    if conn is None:
        return {
            "total_screenings": 0,
            "today_screenings": 0,
            "class_counts": {"No DR": 0, "Mild": 0, "Moderate": 0, "Severe": 0, "Proliferative DR": 0}
        }
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM predictions")
        total = cursor.fetchone()[0] or 0
        
        today_str = datetime.date.today().strftime('%Y-%m-%d')
        if db_type == "sqlite":
            cursor.execute("SELECT COUNT(*) FROM predictions WHERE created_at LIKE ?", (f"{today_str}%",))
        else:
            cursor.execute("SELECT COUNT(*) FROM predictions WHERE DATE(created_at) = CURDATE()")
        today_cnt = cursor.fetchone()[0] or 0
        
        counts = {"No DR": 0, "Mild": 0, "Moderate": 0, "Severe": 0, "Proliferative DR": 0}
        cursor.execute("SELECT prediction_label, COUNT(*) FROM predictions GROUP BY prediction_label")
        for label, count in cursor.fetchall():
            if label in counts:
                counts[label] = count
            elif "No" in label:
                counts["No DR"] += count
            elif "Mild" in label:
                counts["Mild"] += count
            elif "Moderate" in label:
                counts["Moderate"] += count
            elif "Severe" in label:
                counts["Severe"] += count
            elif "Proliferative" in label:
                counts["Proliferative DR"] += count
                
        cursor.close()
        conn.close()
        return {
            "total_screenings": total,
            "today_screenings": today_cnt,
            "class_counts": counts
        }
    except Exception as e:
        print(f"[!] Error building dashboard statistics: {e}")
        return {"total_screenings": 0, "today_screenings": 0, "class_counts": {}}
