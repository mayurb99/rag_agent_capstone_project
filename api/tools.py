from db import get_conn


# ─────────────────────────────────────────────
# APPOINTMENT TOOLS
# ─────────────────────────────────────────────

def book_appointment(patient_id: int, doctor_id: int, appt_date: str, appt_time: str) -> str:
    conn = get_conn()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO appointments (patient_id, doctor_id, appt_date, appt_time, status)
            VALUES (%s, %s, %s, %s, 'scheduled')
            RETURNING id
        """, (patient_id, doctor_id, appt_date, appt_time))
        appt_id = cursor.fetchone()[0]
        conn.commit()

        # Fetch doctor name for confirmation message
        cursor.execute("SELECT name, specialty FROM doctors WHERE id = %s", (doctor_id,))
        doc = cursor.fetchone()
        doc_name = doc[0] if doc else "the doctor"
        specialty = doc[1] if doc else ""

        return (
            f"Appointment booked!\n"
            f"  ID       : #{appt_id}\n"
            f"  Doctor   : {doc_name} ({specialty})\n"
            f"  Date     : {appt_date}\n"
            f"  Time     : {appt_time}\n\n"
            f"Please note your appointment ID #{appt_id} to cancel if needed."
        )
    except Exception as e:
        conn.rollback()
        return f"Failed to book appointment: {str(e)}"
    finally:
        cursor.close()
        conn.close()


def get_appointments(patient_id: int, raw: bool = False):
    """
    raw=False → returns formatted string for display
    raw=True  → returns list of dicts (for cancel flow)
    """
    conn = get_conn()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT a.id, d.name, d.specialty, a.appt_date, a.appt_time, a.status
            FROM appointments a
            JOIN doctors d ON a.doctor_id = d.id
            WHERE a.patient_id = %s AND a.status = 'scheduled'
            ORDER BY a.appt_date, a.appt_time
        """, (patient_id,))
        rows = cursor.fetchall()

        if not rows:
            return [] if raw else "You have no upcoming appointments."

        if raw:
            return [
                {
                    "id": r[0],
                    "doctor_name": r[1],
                    "specialty": r[2],
                    "appt_date": str(r[3]),
                    "appt_time": str(r[4]),
                    "status": r[5],
                }
                for r in rows
            ]

        lines = ["Your upcoming appointments:\n"]
        for r in rows:
            lines.append(
                f"  #{r[0]}  {r[1]} ({r[2]})  |  {r[3]}  {str(r[4])[:5]}  |  {r[5]}"
            )
        return "\n".join(lines)

    finally:
        cursor.close()
        conn.close()


def cancel_appointment(appointment_id: int, patient_id: int) -> str:
    """Soft-cancel: sets status to cancelled. Scoped to patient_id for safety."""
    conn = get_conn()
    cursor = conn.cursor()
    try:
        # Verify appointment belongs to this patient
        cursor.execute("""
            SELECT a.id, d.name, a.appt_date, a.appt_time
            FROM appointments a
            JOIN doctors d ON a.doctor_id = d.id
            WHERE a.id = %s AND a.patient_id = %s AND a.status = 'scheduled'
        """, (appointment_id, patient_id))
        row = cursor.fetchone()

        if not row:
            return (
                f"Appointment #{appointment_id} not found or already cancelled. "
                "Please check the ID and try again."
            )

        cursor.execute("""
            UPDATE appointments SET status = 'cancelled'
            WHERE id = %s AND patient_id = %s
        """, (appointment_id, patient_id))
        conn.commit()

        return (
            f"Appointment #{row[0]} cancelled.\n"
            f"  Doctor : {row[1]}\n"
            f"  Date   : {row[2]}  {str(row[3])[:5]}"
        )
    except Exception as e:
        conn.rollback()
        return f"Failed to cancel appointment: {str(e)}"
    finally:
        cursor.close()
        conn.close()


# ─────────────────────────────────────────────
# DOCTOR TOOLS
# ─────────────────────────────────────────────

def get_doctors(specialty: str = None) -> list:
    """Return list of doctors, optionally filtered by specialty."""
    conn = get_conn()
    cursor = conn.cursor()
    try:
        if specialty:
            cursor.execute("""
                SELECT id, name, specialty, available_days, available_slots
                FROM doctors
                WHERE LOWER(specialty) ILIKE %s
            """, (f"%{specialty.lower()}%",))
        else:
            cursor.execute(
                "SELECT id, name, specialty, available_days, available_slots FROM doctors"
            )
        rows = cursor.fetchall()
        return [
            {
                "id": r[0],
                "name": r[1],
                "specialty": r[2],
                "available_days": r[3],
                "available_slots": r[4],
            }
            for r in rows
        ]
    finally:
        cursor.close()
        conn.close()
