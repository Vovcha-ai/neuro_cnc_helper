import sqlite3
import datetime
import json


def init_db():
    conn = sqlite3.connect('data/predictions.db')
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            predicted_class INTEGER,
            fault_name TEXT,
            confidence REAL,
            symptoms TEXT,
            correct_feedback TEXT
        )
    ''')

    conn.commit()
    conn.close()


def save_prediction(pred_class, fault_name, confidence, symptoms):
    conn = sqlite3.connect('data/predictions.db')
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO predictions (timestamp, predicted_class, fault_name, confidence, symptoms)
        VALUES (?, ?, ?, ?, ?)
    ''', (datetime.datetime.now(), pred_class, fault_name, confidence, json.dumps(symptoms)))

    conn.commit()
    conn.close()


def get_statistics():
    conn = sqlite3.connect('data/predictions.db')
    cursor = conn.cursor()

    cursor.execute('SELECT COUNT(*) FROM predictions')
    total = cursor.fetchone()[0]

    cursor.execute('''
        SELECT fault_name, COUNT(*) as count 
        FROM predictions 
        GROUP BY fault_name 
        ORDER BY count DESC 
        LIMIT 10
    ''')
    top_faults = cursor.fetchall()

    conn.close()
    return total, top_faults


def update_feedback(pred_id, correct_fault):
    conn = sqlite3.connect('data/predictions.db')
    cursor = conn.cursor()

    cursor.execute('''
        UPDATE predictions 
        SET correct_feedback = ? 
        WHERE id = ?
    ''', (correct_fault, pred_id))

    conn.commit()
    conn.close()


init_db()