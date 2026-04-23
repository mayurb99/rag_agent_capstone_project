import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()


def get_conn():
    """Return a new psycopg2 connection. Always close after use."""
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        port=int(os.getenv("DB_PORT", 5432)),
    )


def init_db():
    """Create tables if they don't exist and seed initial data."""
    conn = get_conn()
    cursor = conn.cursor()

    # ── PATIENTS ──────────────────────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            id         SERIAL PRIMARY KEY,
            name       TEXT NOT NULL,
            email      TEXT,
            phone      TEXT,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)

    # ── DOCTORS ───────────────────────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS doctors (
            id               SERIAL PRIMARY KEY,
            name             TEXT NOT NULL,
            specialty        TEXT NOT NULL,
            available_days   TEXT,
            available_slots  TEXT
        )
    """)

    # ── APPOINTMENTS ──────────────────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS appointments (
            id         SERIAL PRIMARY KEY,
            patient_id INTEGER REFERENCES patients(id),
            doctor_id  INTEGER REFERENCES doctors(id),
            appt_date  DATE NOT NULL,
            appt_time  TIME NOT NULL,
            status     TEXT DEFAULT 'scheduled',
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)

    # ── SEED DEMO PATIENT (demo_user) ────────────────────────────────────
    cursor.execute("SELECT id FROM patients WHERE name = 'demo_user'")
    if not cursor.fetchone():
        cursor.execute("""
            INSERT INTO patients (name, email, phone)
            VALUES ('demo_user', 'demo@clinic.com', '9999999999')
        """)
        print("Seeded demo_user patient.")

    # ── SEED DOCTORS ─────────────────────────────────────────────────────
    cursor.execute("SELECT COUNT(*) FROM doctors")
    count = cursor.fetchone()[0]
    if count == 0:
        doctors = [
            ("Dr. Sharma",  "General",          "Mon,Tue,Wed",     "09:00,10:00,11:00,14:00,15:00"),
            ("Dr. Mehta",   "Gastroenterology",  "Thu,Fri",         "10:00,11:00,14:00,15:00,16:00"),
            ("Dr. Kapoor",  "Cardiology",        "Mon,Wed,Fri",     "09:00,10:00,13:00,14:00"),
            ("Dr. Rao",     "Orthopedics",       "Tue,Thu",         "10:00,11:00,15:00,16:00"),
            ("Dr. Nair",    "Neurology",         "Mon,Tue,Thu",     "09:00,11:00,14:00"),
            ("Dr. Gupta",   "Dermatology",       "Wed,Fri",         "10:00,11:00,12:00,15:00"),
            ("Dr. Pillai",  "Pediatrics",        "Mon,Tue,Wed,Thu", "09:00,10:00,11:00,14:00"),
        ]
        cursor.executemany("""
            INSERT INTO doctors (name, specialty, available_days, available_slots)
            VALUES (%s, %s, %s, %s)
        """, doctors)
        print(f"Seeded {len(doctors)} doctors.")

    conn.commit()
    cursor.close()
    conn.close()
    print("Database initialised.")


if __name__ == "__main__":
    init_db()
