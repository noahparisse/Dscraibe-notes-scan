"""
Database operations module for note metadata and event tracking.

This module provides all database interaction functions including:
- Schema initialization and migration
- Note insertion with automatic event grouping
- Similarity-based deduplication (text and image)
- Query functions for notes, threads, and events
- Administrative functions (clear, delete)
"""

import json
import os
import re
import sqlite3
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add repository root to sys.path for internal imports
REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)

from logger_config import setup_logger
from src.image_similarity.image_comparison import isSimilar
from src.ner.compare_entities import same_event
from src.utils.text_utils import compute_diff

logger = setup_logger(__name__)

# =============================================================================
# Configuration Constants

# Database path (can be overridden via environment variable)
DB_PATH = os.environ.get(
    "RTE_DB_PATH", os.path.join(REPO_PATH, "data/db/notes.sqlite")
)

# Text similarity threshold for note matching
# Range: 0.0-1.0 where 1.0 = identical text required
# Used by find_similar_note() to determine if two transcriptions are from the same physical note
NOTE_TEXT_SIMILARITY_THRESHOLD: float = 0.3

# Image comparison - number of recent notes to check
# When checking if an image is a duplicate, compare against the last N notes
# Higher values = more thorough duplicate detection but slower processing
IMAGE_COMPARISON_LIMIT: int = 20

# Query result limits
DEFAULT_LIST_LIMIT: int = 20  # Default number of notes returned by list_notes()
DEFAULT_THREAD_LIMIT: int = 50  # Default limit for note thread queries

# =============================================================================


def _resolve_db_path(db_path: Optional[str]) -> str:
    """
    Resolve database path, using default if None provided.

    Args:
        db_path: Optional path to database file.

    Returns:
        Resolved database path (either provided path or default DB_PATH).
    """
    return db_path or DB_PATH


def ensure_db(db_path: str = DB_PATH):
    """
    Initialize database schema if it doesn't exist.

    Creates the notes_meta table with all required columns for storing:
    - Note metadata (timestamps, IDs, confidence scores)
    - Transcription data (raw OCR, cleaned text, diff text)
    - Image paths for photo-based notes
    - Extracted entities (9 entity types as JSON arrays)
    - Event grouping (evenement_id for linking related notes)

    Also creates necessary indexes for performance optimization.

    Args:
        db_path: Path to the SQLite database file.

    Side Effects:
        - Creates database file if it doesn't exist
        - Creates parent directories if needed
        - Creates notes_meta table and indexes
    """
    db_path = _resolve_db_path(db_path)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    logger.debug(f"Ensuring database exists at: {db_path}")

    con = sqlite3.connect(db_path)
    con.execute("""
    CREATE TABLE IF NOT EXISTS notes_meta (
        id                      INTEGER PRIMARY KEY AUTOINCREMENT,
        ts                      INTEGER NOT NULL,
        note_id                 TEXT,
        transcription_brute     TEXT,
        transcription_clean     TEXT,
        texte_ajoute            TEXT,
        confidence_score        REAL DEFAULT 0.0,    
        img_path_proc           TEXT,
        raw_json                TEXT,

        -- Colonnes pour entités extraites (JSON arrays)
        entite_GEO                     TEXT,
        entite_ACTOR                   TEXT,
        entite_DATETIME                TEXT,
        entite_EVENT                   TEXT,
        entite_INFRASTRUCTURE          TEXT,
        entite_OPERATING_CONTEXT       TEXT,
        entite_PHONE_NUMBER            TEXT,
        entite_ELECTRICAL_VALUE        TEXT,
        entite_ABBREVIATION_UNKNOWN    TEXT,

        -- Colonne évènement pour regroupement
        evenement_id            TEXT
    );
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_notes_meta_ts ON notes_meta(ts);")
    con.commit()
    con.close()

    logger.debug("Database schema initialized successfully")


def insert_note_meta(
    meta: dict, img_path_proc: Optional[str] = None, db_path: str = DB_PATH
) -> int:
    """
    Insert a new note metadata record into the database.

    This function performs several operations:
    1. Extract entities from the metadata
    2. Find or create an event ID (evenement_id) for grouping related notes
    3. Insert the complete record with all fields
    4. Return the auto-generated database ID

    Args:
        meta: Dictionary containing all note metadata fields:
            - note_id: Unique identifier for this note thread
            - transcription_brute: Raw OCR/ASR output
            - transcription_clean: Normalized transcription
            - texte_ajoute: Diff text (new content only)
            - confidence_score: OCR/ASR confidence (0.0-1.0)
            - raw_json: Additional metadata as JSON string
            - entite_*: JSON arrays for each entity type
        img_path_proc: Optional path to processed image file.
        db_path: Path to the SQLite database file.

    Returns:
        The auto-incremented database ID of the inserted record.

    Side Effects:
        - May create a new event ID if no matching event is found
        - Updates event ID counter in database

    Example:
        >>> meta = {
        ...     "note_id": "TEXT-1",
        ...     "transcription_clean": "Appel Jean 14h",
        ...     "entite_ACTOR": '["Jean"]',
        ...     # ... other fields
        ... }
        >>> record_id = insert_note_meta(meta)
        >>> print(f"Inserted record {record_id}")
    """
    ensure_db(db_path)
    now = int(time.time())

    logger.debug(f"Inserting note metadata for note_id={meta.get('note_id')}")

    # Parse entity JSON arrays from metadata
    entities_new = {
        "GEO": json.loads(meta.get("entite_GEO") or "[]"),
        "DATETIME": json.loads(meta.get("entite_DATETIME") or "[]"),
        "EVENT": json.loads(meta.get("entite_EVENT") or "[]"),
        "ACTOR": json.loads(meta.get("entite_ACTOR") or "[]"),
        "INFRASTRUCTURE": json.loads(meta.get("entite_INFRASTRUCTURE") or "[]"),
        "OPERATING_CONTEXT": json.loads(meta.get("entite_OPERATING_CONTEXT") or "[]"),
        "PHONE_NUMBER": json.loads(meta.get("entite_PHONE_NUMBER") or "[]"),
        "ELECTRICAL_VALUE": json.loads(meta.get("entite_ELECTRICAL_VALUE") or "[]"),
        "ABBREVIATION_UNKNOWN": json.loads(
            meta.get("entite_ABBREVIATION_UNKNOWN") or "[]"
        ),
    }

    # Find existing event or create new one
    evenement_id = find_existing_event_id(entities_new, db_path)
    if evenement_id is None:
        last_id = get_last_event_number_from_db(db_path)
        nouvel_id = last_id + 1
        evenement_id = f"EVT-{nouvel_id}"
        logger.info(f"Creating new event: {evenement_id}")
    else:
        logger.debug(f"Linking to existing event: {evenement_id}")

    # Prepare row data for insertion
    row = (
        now,
        meta.get("note_id"),
        meta.get("transcription_brute"),
        meta.get("transcription_clean"),
        meta.get("texte_ajoute"),
        meta.get("confidence_score"),
        img_path_proc,
        meta.get("raw_json") or json.dumps(meta, ensure_ascii=False),
        meta.get("entite_GEO"),
        meta.get("entite_ACTOR"),
        meta.get("entite_DATETIME"),
        meta.get("entite_EVENT"),
        meta.get("entite_INFRASTRUCTURE"),
        meta.get("entite_OPERATING_CONTEXT"),
        meta.get("entite_PHONE_NUMBER"),
        meta.get("entite_ELECTRICAL_VALUE"),
        meta.get("entite_ABBREVIATION_UNKNOWN"),
        evenement_id,
    )

    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute(
        """
        INSERT INTO notes_meta (
            ts, note_id, transcription_brute, transcription_clean, texte_ajoute, confidence_score,
            img_path_proc, raw_json,
            entite_GEO, entite_ACTOR, entite_DATETIME, entite_EVENT,
            entite_INFRASTRUCTURE, entite_OPERATING_CONTEXT,
            entite_PHONE_NUMBER, entite_ELECTRICAL_VALUE, entite_ABBREVIATION_UNKNOWN,
            evenement_id
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        row,
    )

    con.commit()
    new_id = cur.lastrowid
    con.close()

    logger.info(
        f"Successfully inserted note metadata with id={new_id}, evenement_id={evenement_id}"
    )
    return new_id


def list_notes(limit: int = DEFAULT_LIST_LIMIT, db_path: str = DB_PATH) -> List[Dict]:
    """
    Retrieve the most recent notes from the database.

    Args:
        limit: Maximum number of notes to return.
        db_path: Path to the SQLite database file.

    Returns:
        List of note dictionaries, sorted by timestamp descending (newest first).
        Each dictionary contains all fields from the notes_meta table.

    Example:
        >>> recent_notes = list_notes(limit=10)
        >>> for note in recent_notes:
        ...     print(f"{note['note_id']}: {note['transcription_clean']}")
    """
    ensure_db(db_path)
    logger.debug(f"Listing {limit} most recent notes")

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute(
        """
        SELECT id, ts, note_id, transcription_brute, transcription_clean, texte_ajoute, img_path_proc,
               confidence_score,
               entite_GEO, entite_ACTOR, entite_DATETIME, entite_EVENT,
               entite_INFRASTRUCTURE, entite_OPERATING_CONTEXT,
               entite_PHONE_NUMBER, entite_ELECTRICAL_VALUE, entite_ABBREVIATION_UNKNOWN,
               evenement_id
        FROM notes_meta
        ORDER BY ts DESC
        LIMIT ?
    """,
        (limit,),
    )
    rows = [dict(r) for r in cur.fetchall()]
    con.close()

    logger.debug(f"Retrieved {len(rows)} notes from database")
    return rows


def get_last_note_text(db_path: str = DB_PATH) -> Tuple[Optional[str], Optional[int]]:
    """
    Retrieve the cleaned text and ID of the most recent note.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        A tuple containing:
        - transcription_clean: Cleaned text of the last note (or None if no notes)
        - id: Database ID of the last note (or None if no notes)

    Example:
        >>> text, note_id = get_last_note_text()
        >>> if text:
        ...     print(f"Last note (id={note_id}): {text}")
    """
    ensure_db(db_path)
    logger.debug("Retrieving last note text")

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute(
        """
        SELECT id, transcription_clean FROM notes_meta
        ORDER BY ts DESC
        LIMIT 1
    """
    )
    row = cur.fetchone()
    con.close()

    if row and row["transcription_clean"]:
        logger.debug(f"Retrieved last note: id={row['id']}, length={len(row['transcription_clean'])} chars")
        return row["transcription_clean"], row["id"]

    logger.debug("No notes found in database")
    return None, None


def is_same_note(
    clean_text: str, db_path: str = DB_PATH, threshold: float = 0.7
) -> Optional[int]:
    """
    Check if the provided text matches the last note in the database (legacy function).

    This function is deprecated in favor of find_similar_note() which compares
    against all notes, not just the most recent one.

    Args:
        clean_text: Cleaned transcription text to compare.
        db_path: Path to the SQLite database file.
        threshold: Minimum similarity ratio (0.0-1.0) to consider a match.

    Returns:
        The database ID of the last note if similarity exceeds threshold, otherwise None.

    Note:
        This function truncates both texts to the length of the shorter one before
        comparing, which may not be ideal for all use cases.
    """
    ensure_db(db_path)
    logger.debug(f"Checking if text matches last note (threshold={threshold})")

    last_summary, last_id = get_last_note_text(db_path)
    if last_summary is None:
        logger.debug("No previous note to compare against")
        return None

    min_len = min(len(clean_text), len(last_summary))
    a_trunc = clean_text[:min_len]
    b_trunc = last_summary[:min_len]
    ratio = SequenceMatcher(None, a_trunc, b_trunc).ratio()

    logger.debug(f"Similarity with last note: {ratio:.3f}")

    if ratio >= threshold:
        logger.info(f"Text matches last note (id={last_id}, similarity={ratio:.3f})")
        return last_id

    return None


def get_last_image_for_notes(db_path: str = DB_PATH) -> Dict[str, str]:
    """
    Retrieve the most recent image path for each note thread.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        Dictionary mapping note_id to img_path_proc for the latest image of each note.
        Only includes notes with non-null note_id and img_path_proc.

    Example:
        >>> images = get_last_image_for_notes()
        >>> for note_id, img_path in images.items():
        ...     print(f"{note_id}: {img_path}")
    """
    ensure_db(db_path)
    logger.debug("Retrieving last image for each note thread")

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute(
        """
        SELECT note_id, img_path_proc, MAX(ts) as ts
        FROM notes_meta
        WHERE note_id IS NOT NULL
        GROUP BY note_id
    """
    )
    result = {}
    for row in cur.fetchall():
        if row["note_id"] and row["img_path_proc"]:
            result[row["note_id"]] = row["img_path_proc"]
    con.close()

    logger.debug(f"Retrieved images for {len(result)} note threads")
    return result


def get_last_text_for_notes(db_path: str = DB_PATH) -> Dict[str, str]:
    """
    Retrieve the most recent cleaned text for each note thread.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        Dictionary mapping note_id to transcription_clean for the latest text of each note.
        Only includes notes with non-null note_id and transcription_clean.

    Example:
        >>> texts = get_last_text_for_notes()
        >>> for note_id, text in texts.items():
        ...     print(f"{note_id}: {text[:50]}...")
    """
    ensure_db(db_path)
    logger.debug("Retrieving last text for each note thread")

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute(
        """
        SELECT note_id, transcription_clean, MAX(ts) as ts
        FROM notes_meta
        WHERE note_id IS NOT NULL
        GROUP BY note_id
    """
    )
    result = {}
    for row in cur.fetchall():
        if row["note_id"] and row["transcription_clean"]:
            result[row["note_id"]] = row["transcription_clean"]
    con.close()

    logger.debug(f"Retrieved texts for {len(result)} note threads")
    return result


def get_last_notes(db_path: str = DB_PATH) -> List[str]:
    """
    Retrieve image paths for the most recent notes (one per note thread).

    This function returns the latest image for each unique note_id, limited to
    the IMAGE_COMPARISON_LIMIT most recent notes. Used primarily for image
    similarity checking during deduplication.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        List of image paths (strings) for the most recent notes, sorted by timestamp descending.

    Example:
        >>> recent_images = get_last_notes()
        >>> print(f"Found {len(recent_images)} recent note images")
    """
    ensure_db(db_path)
    logger.debug(f"Retrieving last {IMAGE_COMPARISON_LIMIT} note images")

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute(
        f"""
        SELECT DISTINCT n1.img_path_proc
        FROM notes_meta AS n1
        INNER JOIN (
            SELECT note_id, MAX(ts) AS max_ts
            FROM notes_meta
            WHERE note_id IS NOT NULL
            GROUP BY note_id
        ) AS n2
        ON n1.note_id = n2.note_id AND n1.ts = n2.max_ts
        ORDER BY n1.ts DESC
        LIMIT {IMAGE_COMPARISON_LIMIT};
    """
    )
    result = []
    for row in cur.fetchall():
        if row["img_path_proc"]:
            result.append(row["img_path_proc"])
    con.close()

    logger.debug(f"Retrieved {len(result)} recent note images")
    return result


def find_similar_note(
    clean_text: str,
    db_path: str = DB_PATH,
    threshold: float = NOTE_TEXT_SIMILARITY_THRESHOLD,
) -> Optional[str]:
    """
    Search for an existing note with similar text content.

    Compares the provided text against the most recent version of all note threads
    in the database. Returns the note_id of the first match exceeding the similarity
    threshold.

    Args:
        clean_text: Cleaned transcription text to compare.
        db_path: Path to the SQLite database file.
        threshold: Minimum similarity ratio (0.0-1.0) to consider a match.

    Returns:
        The note_id of a similar note if found, otherwise None.

    Note:
        This function truncates both texts to the length of the shorter one before
        comparing. For very short texts, this may lead to false positives.

    Example:
        >>> similar = find_similar_note("Appel Jean 14h maintenance")
        >>> if similar:
        ...     print(f"Found similar note: {similar}")
        ... else:
        ...     print("This is a new note")
    """
    logger.debug(f"Searching for similar notes (threshold={threshold})")

    last_texts = get_last_text_for_notes(db_path)
    for note_id, last_summary in last_texts.items():
        min_len = min(len(clean_text), len(last_summary))
        a_trunc = clean_text[:min_len]
        b_trunc = last_summary[:min_len]
        ratio = SequenceMatcher(None, a_trunc, b_trunc).ratio()

        logger.debug(f"Comparing with {note_id}: similarity={ratio:.3f}")

        if ratio >= threshold:
            logger.info(f"Found similar note: {note_id} (similarity={ratio:.3f})")
            return note_id

    logger.debug("No similar note found")
    return None


def find_similar_image(new_img: Path, db_path: str = DB_PATH) -> Optional[str]:
    """
    Search for an existing note with a visually similar image.

    Compares the provided image against the most recent images from recent note threads
    using perceptual hashing. Returns the path of the first matching image if found.

    Args:
        new_img: Path to the new image to compare.
        db_path: Path to the SQLite database file.

    Returns:
        The image path of a similar note if found, otherwise None.

    Note:
        This function only compares against the IMAGE_COMPARISON_LIMIT most recent
        notes for performance reasons. Very old notes may not be checked.

    Example:
        >>> from pathlib import Path
        >>> new_image = Path("tmp/paper/detection_20250101-120000.jpg")
        >>> similar = find_similar_image(new_image)
        >>> if similar:
        ...     print(f"Duplicate image detected: {similar}")
    """
    logger.debug(f"Searching for similar images to: {new_img}")

    last_notes = get_last_notes(db_path)
    for i, path in enumerate(last_notes):
        logger.debug(f"Comparing with image {i+1}/{len(last_notes)}: {path}")
        if isSimilar(path, new_img):
            logger.info(f"Found similar image: {path}")
            return path

    logger.debug("No similar image found")
    return None


def get_added_text(
    old_text: str, new_text: str, minor_change_threshold: float = 0.90
) -> str:
    """
    Compute human-readable diff between two texts (legacy wrapper).

    This is a backward-compatibility wrapper around compute_diff() that only
    returns the human-readable string output, discarding the JSON structure.

    Args:
        old_text: Original text.
        new_text: Updated text.
        minor_change_threshold: Similarity threshold below which changes are reported.

    Returns:
        Human-readable diff string with line-level operations marked with +, ~, -.

    Note:
        For new code, prefer using compute_diff() directly which returns both
        human-readable and structured JSON formats.
    """
    logger.debug("Computing text diff (legacy get_added_text wrapper)")
    human, _ = compute_diff(old_text, new_text, minor_change_threshold=minor_change_threshold)
    return human


def get_last_note_meta(db_path: str = DB_PATH) -> Optional[Dict]:
    """
    Retrieve the complete metadata record for the most recent note.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        Dictionary containing all fields from the notes_meta table for the last note,
        or None if no notes exist.

    Example:
        >>> last_note = get_last_note_meta()
        >>> if last_note:
        ...     print(f"Last note ID: {last_note['note_id']}")
        ...     print(f"Event ID: {last_note['evenement_id']}")
    """
    ensure_db(db_path)
    logger.debug("Retrieving last note metadata")

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute(
        """
        SELECT * FROM notes_meta
        ORDER BY ts DESC
        LIMIT 1
    """
    )
    row = cur.fetchone()
    con.close()

    if row:
        logger.debug(f"Retrieved last note: id={row['id']}, note_id={row['note_id']}")
        return dict(row)

    logger.debug("No notes found in database")
    return None


def clear_notes_meta(db_path: str = DB_PATH):
    """
    Delete all records from the notes_meta table and reset auto-increment counter.

    This is a destructive operation that permanently removes all note data.
    Use with caution, typically only for testing or database resets.

    Args:
        db_path: Path to the SQLite database file.

    Side Effects:
        - Deletes all rows from notes_meta table
        - Resets the AUTOINCREMENT counter to 0
        - Prints confirmation message to stdout

    Example:
        >>> clear_notes_meta()
        La base de données a été vidée.
    """
    logger.warning(f"Clearing entire database at: {db_path}")

    if os.path.exists(db_path):
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        cur.execute("DELETE FROM notes_meta;")
        cur.execute("DELETE FROM sqlite_sequence WHERE name='notes_meta';")
        con.commit()
        con.close()
        logger.info("Database cleared successfully")
        print("La base de données a été vidée.")
    else:
        logger.warning(f"Database not found at: {db_path}")
        print("Pas de base de données trouvée à l'adresse :", db_path)


def delete_entry_by_id(entry_id: int, db_path: str = DB_PATH) -> int:
    """
    Delete a single note entry by its database ID.

    Args:
        entry_id: The database ID (primary key) of the entry to delete.
        db_path: Path to the SQLite database file.

    Returns:
        Number of rows deleted (0 if entry not found, 1 if deleted successfully).

    Example:
        >>> deleted = delete_entry_by_id(42)
        >>> if deleted:
        ...     print("Entry deleted successfully")
        ... else:
        ...     print("Entry not found")
    """
    ensure_db(db_path)
    logger.info(f"Deleting entry by id: {entry_id}")

    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("DELETE FROM notes_meta WHERE id = ?", (entry_id,))
    deleted = cur.rowcount
    con.commit()
    con.close()

    logger.info(f"Deleted {deleted} entry(ies)")
    return deleted


def delete_thread_by_note_id(note_id: str, db_path: str = DB_PATH) -> int:
    """
    Delete all entries associated with a specific note thread.

    This removes the entire history of a note, including all versions and updates.

    Args:
        note_id: The note_id (thread identifier) to delete.
        db_path: Path to the SQLite database file.

    Returns:
        Number of rows deleted (0 if note_id not found, ≥1 if entries deleted).

    Example:
        >>> deleted = delete_thread_by_note_id("TEXT-5")
        >>> print(f"Deleted {deleted} entries from thread TEXT-5")
    """
    ensure_db(db_path)
    logger.info(f"Deleting entire thread for note_id: {note_id}")

    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("DELETE FROM notes_meta WHERE note_id = ?", (note_id,))
    deleted = cur.rowcount
    con.commit()
    con.close()

    logger.info(f"Deleted {deleted} entry(ies) from thread {note_id}")
    return deleted


def list_notes_by_note_id(
    note_id: str, db_path: str = DB_PATH, limit: int = DEFAULT_THREAD_LIMIT
) -> List[Dict]:
    """
    Retrieve all entries for a specific note thread, sorted by timestamp.

    Returns the complete history of a note, including all versions and updates.

    Args:
        note_id: The note_id (thread identifier) to query.
        db_path: Path to the SQLite database file.
        limit: Maximum number of entries to return.

    Returns:
        List of note dictionaries, sorted by timestamp descending (newest first).

    Example:
        >>> thread = list_notes_by_note_id("TEXT-5")
        >>> print(f"Note TEXT-5 has {len(thread)} versions")
        >>> for version in thread:
        ...     print(f"  - {version['ts']}: {version['texte_ajoute']}")
    """
    ensure_db(db_path)
    logger.debug(f"Listing entries for note_id={note_id} (limit={limit})")

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute(
        """
        SELECT id, ts, note_id, transcription_brute, transcription_clean, texte_ajoute, img_path_proc,
               confidence_score,
               entite_GEO, entite_ACTOR, entite_DATETIME, entite_EVENT,
               entite_INFRASTRUCTURE, entite_OPERATING_CONTEXT,
               entite_PHONE_NUMBER, entite_ELECTRICAL_VALUE, entite_ABBREVIATION_UNKNOWN,
               evenement_id
        FROM notes_meta
        WHERE note_id = ?
        ORDER BY ts DESC
        LIMIT ?
    """,
        (note_id, limit),
    )
    rows = [dict(r) for r in cur.fetchall()]
    con.close()

    logger.debug(f"Retrieved {len(rows)} entries for note_id={note_id}")
    return rows


def list_notes_by_evenement_id(
    evenement_id: str, db_path: str = DB_PATH, limit: int = DEFAULT_THREAD_LIMIT
) -> List[Dict]:
    """
    Retrieve all notes associated with a specific event.

    Returns all notes that have been grouped under the same event ID based on
    entity similarity (location, time, actors, etc.).

    Args:
        evenement_id: The event ID to query (format: "EVT-N").
        db_path: Path to the SQLite database file.
        limit: Maximum number of notes to return.

    Returns:
        List of note dictionaries, sorted by timestamp descending (newest first).

    Example:
        >>> event_notes = list_notes_by_evenement_id("EVT-42")
        >>> print(f"Event EVT-42 has {len(event_notes)} related notes")
        >>> for note in event_notes:
        ...     print(f"  - {note['note_id']}: {note['transcription_clean']}")
    """
    ensure_db(db_path)
    logger.debug(f"Listing notes for evenement_id={evenement_id} (limit={limit})")

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute(
        """
        SELECT id, ts, note_id, transcription_brute, transcription_clean, texte_ajoute, img_path_proc,
               confidence_score,
               entite_GEO, entite_ACTOR, entite_DATETIME, entite_EVENT,
               entite_INFRASTRUCTURE, entite_OPERATING_CONTEXT,
               entite_PHONE_NUMBER, entite_ELECTRICAL_VALUE, entite_ABBREVIATION_UNKNOWN,
               evenement_id
        FROM notes_meta
        WHERE evenement_id = ?
        ORDER BY ts DESC
        LIMIT ?
    """,
        (evenement_id, limit),
    )
    rows = [dict(r) for r in cur.fetchall()]
    con.close()

    logger.debug(f"Retrieved {len(rows)} notes for evenement_id={evenement_id}")
    return rows


def find_existing_event_id(entities_new: Dict, db_path: str = DB_PATH) -> Optional[str]:
    """
    Search for an existing event that matches the provided entities.

    Compares the entity set from a new note against all existing events in the
    database using the same_event() function. Returns the event ID if a match
    is found, enabling automatic grouping of related notes.

    Args:
        entities_new: Dictionary of entity arrays for the new note, with keys:
            - GEO: List of locations
            - DATETIME: List of timestamps
            - EVENT: List of event keywords
            - ACTOR: List of person names
            - INFRASTRUCTURE: List of equipment/facility names
            - OPERATING_CONTEXT: List of operational states
            - PHONE_NUMBER: List of phone numbers
            - ELECTRICAL_VALUE: List of voltage/current measurements
            - ABBREVIATION_UNKNOWN: List of unrecognized abbreviations
        db_path: Path to the SQLite database file.

    Returns:
        The evenement_id of a matching event if found, otherwise None.

    Note:
        Entity matching uses fuzzy string similarity and domain-specific rules
        defined in src.ner.compare_entities.same_event().

    Example:
        >>> entities = {
        ...     "GEO": ["Paris", "Gare du Nord"],
        ...     "DATETIME": ["14h"],
        ...     "ACTOR": ["Jean"],
        ...     # ... other entity types
        ... }
        >>> event_id = find_existing_event_id(entities)
        >>> if event_id:
        ...     print(f"Linking to existing event: {event_id}")
        ... else:
        ...     print("Creating new event")
    """
    logger.debug("Searching for existing event matching provided entities")

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute(
        """
        SELECT evenement_id, entite_GEO, entite_DATETIME, entite_EVENT, entite_ACTOR,
               entite_INFRASTRUCTURE, entite_OPERATING_CONTEXT, entite_PHONE_NUMBER,
               entite_ELECTRICAL_VALUE, entite_ABBREVIATION_UNKNOWN
        FROM notes_meta
        WHERE evenement_id IS NOT NULL
    """
    )

    for i, row in enumerate(cur.fetchall()):
        entities_existing = {
            "GEO": json.loads(row["entite_GEO"] or "[]"),
            "DATETIME": json.loads(row["entite_DATETIME"] or "[]"),
            "EVENT": json.loads(row["entite_EVENT"] or "[]"),
            "ACTOR": json.loads(row["entite_ACTOR"] or "[]"),
            "INFRASTRUCTURE": json.loads(row["entite_INFRASTRUCTURE"] or "[]"),
            "OPERATING_CONTEXT": json.loads(row["entite_OPERATING_CONTEXT"] or "[]"),
            "PHONE_NUMBER": json.loads(row["entite_PHONE_NUMBER"] or "[]"),
            "ELECTRICAL_VALUE": json.loads(row["entite_ELECTRICAL_VALUE"] or "[]"),
            "ABBREVIATION_UNKNOWN": json.loads(
                row["entite_ABBREVIATION_UNKNOWN"] or "[]"
            ),
        }

        logger.debug(f"Comparing with event {i+1}: {row['evenement_id']}")

        if same_event(entities_new, entities_existing):
            con.close()
            logger.info(f"Found matching event: {row['evenement_id']}")
            return row["evenement_id"]

    con.close()
    logger.debug("No matching event found")
    return None


def get_last_event_number_from_db(db_path: str = DB_PATH) -> int:
    """
    Retrieve the highest event number currently in the database.

    Parses all evenement_id values (format: "EVT-N") and returns the maximum N found.
    Used to generate the next sequential event ID.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        The highest event number found, or 0 if no events exist.

    Example:
        >>> max_event = get_last_event_number_from_db()
        >>> next_event_id = f"EVT-{max_event + 1}"
        >>> print(f"Next event will be: {next_event_id}")
    """
    logger.debug("Retrieving last event number from database")

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute(
        """
        SELECT DISTINCT evenement_id 
        FROM notes_meta
        WHERE evenement_id IS NOT NULL
    """
    )
    rows = cur.fetchall()
    con.close()

    max_num = 0
    for row in rows:
        evt_id = row["evenement_id"]
        if evt_id:
            match = re.search(r"EVT-(\d+)", evt_id)
            if match:
                num = int(match.group(1))
                if num > max_num:
                    max_num = num

    logger.debug(f"Last event number: {max_num}")
    return max_num