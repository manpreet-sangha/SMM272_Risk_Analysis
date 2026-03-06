"""
SMM272 Risk Analysis — Centralised logging utility.

Usage
-----
Master script (once per run)::

    from logger import setup_run_logger
    log_path = setup_run_logger("smm272_q1")

Each module::

    from logger import get_logger, log_start, log_end
    logger = get_logger("q1_1_download_prices")

    def my_func():
        log_start(logger, "q1_1_download_prices.py")
        logger.info("doing work ...")
        log_end(logger, "q1_1_download_prices.py")

Web-app real-time streaming
---------------------------
The log file is written with an immediate flush after every record so that
a web application can tail-follow the file in real time (e.g. via a
server-sent events endpoint that reads the log line by line).

Log files are saved to:  <project_root>/logs/
File naming convention:  <prefix>_run_YYYYMMDD_HHMMSS.log
"""

import logging
import os
import sys
from datetime import datetime

# ── Paths ──────────────────────────────────────────────────────────────────────
_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR  = os.path.join(_ROOT_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

BANNER = "*" * 54

# ── Internal state ─────────────────────────────────────────────────────────────
_run_log_path = None   # absolute path of the current run's log file


# ── Custom handler ─────────────────────────────────────────────────────────────
class _FlushFileHandler(logging.FileHandler):
    """FileHandler that flushes the buffer after every record.

    This ensures the log file is always current so a web application can
    stream its contents in real time without waiting for the OS I/O buffer.
    """

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()


# ── Public API ─────────────────────────────────────────────────────────────────

def setup_run_logger(prefix: str = "smm272") -> str:
    """Initialise the root logger for a single analysis run.

    Creates a timestamped log file under ``<project_root>/logs/`` and
    attaches both a console handler (stdout) and a flush-on-write file
    handler.  Call this **once** from the master script before any module
    function is invoked.

    Parameters
    ----------
    prefix:
        Short label prepended to the log filename, e.g. ``"smm272_q1"``.

    Returns
    -------
    str
        Absolute path to the log file created for this run.
    """
    global _run_log_path

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{prefix}_run_{timestamp}.log"
    _run_log_path = os.path.join(LOGS_DIR, log_filename)

    # ── Enforce a maximum of 5 log files; delete oldest first ─────────────
    _MAX_LOG_FILES = 5
    existing = sorted(
        [os.path.join(LOGS_DIR, f) for f in os.listdir(LOGS_DIR) if f.endswith(".log")],
        key=os.path.getmtime,
    )
    for old_file in existing[: max(0, len(existing) - _MAX_LOG_FILES + 1)]:
        try:
            os.remove(old_file)
        except OSError:
            pass

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()          # remove any handlers added by third-party libs

    fmt = logging.Formatter(
        "%(asctime)s  %(name)-32s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler — INFO and above
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # File handler — DEBUG and above, flushed after every record
    fh = _FlushFileHandler(_run_log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    return _run_log_path


def get_logger(name: str) -> logging.Logger:
    """Return a named child logger that inherits the root handlers.

    Parameters
    ----------
    name:
        Typically the module filename without extension, e.g.
        ``"q1_1_download_prices"``.
    """
    return logging.getLogger(name)


def log_start(logger: logging.Logger, module_filename: str) -> None:
    """Emit a start banner for *module_filename*."""
    logger.info(BANNER)
    logger.info("Start - %s", module_filename)
    logger.info(BANNER)


def log_end(logger: logging.Logger, module_filename: str) -> None:
    """Emit an end banner for *module_filename*."""
    logger.info(BANNER)
    logger.info("End - %s", module_filename)
    logger.info(BANNER)
