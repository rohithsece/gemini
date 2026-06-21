"""List, save, and run Python query files in student_management/queries/."""

import importlib.util
import inspect
import sqlite3
import sys
from pathlib import Path

from student_management.core import BASE_DIR, DB_PATH

QUERIES_DIR = BASE_DIR / "queries"


def list_queries() -> list[str]:
    QUERIES_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(f.name for f in QUERIES_DIR.glob("*.py"))


def _sanitize_filename(filename: str) -> str:
    name = "".join(c for c in filename if c.isalnum() or c in "._-")
    if not name.endswith(".py"):
        name += ".py"
    return name


def save_query(filename: str, code: str) -> Path:
    QUERIES_DIR.mkdir(parents=True, exist_ok=True)
    path = QUERIES_DIR / _sanitize_filename(filename)
    path.write_text(code, encoding="utf-8")
    return path


def run_query(filename: str) -> str:
    safe = _sanitize_filename(filename)
    file_path = QUERIES_DIR / safe
    if not file_path.exists():
        raise FileNotFoundError(safe)

    queries_parent = str(QUERIES_DIR.parent)
    if queries_parent not in sys.path:
        sys.path.insert(0, queries_parent)

    mod_name = f"student_management.queries.{safe[:-3]}"
    spec = importlib.util.spec_from_file_location(mod_name, str(file_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)

    mod_name = module.__name__
    funcs = [
        name
        for name, val in module.__dict__.items()
        if callable(val)
        and not name.startswith("_")
        and getattr(val, "__module__", "") == mod_name
    ]
    target_func = None
    for name in ("main", "run", safe[:-3]):
        if name in funcs:
            target_func = getattr(module, name)
            break
    if not target_func and funcs:
        target_func = getattr(module, funcs[0])

    if not target_func:
        return "Module loaded. No callable function found to run."

    sig = inspect.signature(target_func)
    params = list(sig.parameters.keys())
    args = {}
    if "conn" in params:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        args["conn"] = conn
    if "student_id" in params:
        args["student_id"] = 10045
    if "course" in params:
        args["course"] = "Biology 101"
    if "grade" in params:
        args["grade"] = "A-"
    if "record_id" in params:
        args["record_id"] = 1
    if "new_grade" in params:
        args["new_grade"] = "A+"

    try:
        result = target_func(**args)
        if isinstance(result, list):
            formatted_list = []
            for item in result:
                if isinstance(item, sqlite3.Row):
                    formatted_list.append(dict(item))
                elif hasattr(item, "keys"):  # check if it behaves like a Row/dict
                    try:
                        formatted_list.append(dict(item))
                    except (ValueError, TypeError):
                        formatted_list.append(item)
                else:
                    formatted_list.append(item)
            result = formatted_list
        elif isinstance(result, sqlite3.Row):
            result = dict(result)
        elif hasattr(result, "keys"):
            try:
                result = dict(result)
            except (ValueError, TypeError):
                pass
        return f"Executed {target_func.__name__}(): {result}"
    finally:
        if "conn" in args:
            args["conn"].close()
