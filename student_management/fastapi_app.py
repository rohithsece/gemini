"""FastAPI CRUD API for student records."""

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware

from student_management.core import (
    SECRET_KEY,
    create_demo_user,
    get_connection,
    init_db,
    login_student,
)
from student_management.document_store import save_document
from student_management.query_runner import list_queries, run_query, save_query

import os
import re
from student_management.query_codegen import (
    is_crud_query,
    build_crud_prompt,
    extract_python_code,
    infer_filename,
    save_query_file,
)
from RAG_groq import answer_with_groq

class LoginBody(BaseModel):
    student_id: int
    password: str


class RecordCreate(BaseModel):
    course: str
    grade: str


class RecordUpdate(BaseModel):
    grade: str


class QuerySave(BaseModel):
    filename: str
    code: str


class QueryGenerateBody(BaseModel):
    description: str


def _student_id(request: Request) -> int:
    sid = request.session.get("student_id")
    if sid is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    return int(sid)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    create_demo_user()
    yield


app = FastAPI(
    title="Student Management API",
    description="CRUD API for student academic records (FastAPI)",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)


@app.get("/")
def root():
    return {
        "service": "Student Management API",
        "docs": "/docs",
        "queries": "/queries",
        "explorer": "/query-explorer",
        "demo_login": {"student_id": 10045, "password": "SecurePassword123"},
    }


@app.get("/queries")
def api_list_queries():
    return list_queries()


@app.post("/queries")
def api_save_query(body: QuerySave):
    path = save_query(body.filename, body.code)
    return {"message": f"Saved {path.name} successfully!", "path": str(path)}


@app.post("/queries/generate")
def api_generate_query(body: QueryGenerateBody):
    description = body.description.strip()
    if not description:
        raise HTTPException(status_code=400, detail="Missing 'description' field")
    try:
        crud = is_crud_query(description)
        if crud:
            prompt = build_crud_prompt(description)
        else:
            prompt = (
                "Generate the complete source code for the following request. "
                "Return ONLY a markdown fenced code block with the appropriate language tag. "
                f"Do not include any explanation or surrounding text. Request: {description}"
            )
        model = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
        api_key = os.environ.get("GROQ_API_KEY", "").strip()
        if not api_key:
            raise HTTPException(status_code=500, detail="GROQ_API_KEY is not configured in backend environment.")

        raw_answer, _ = answer_with_groq(
            model=model,
            api_key=api_key,
            query=prompt,
            context=""
        )

        if crud:
            code = extract_python_code(raw_answer)
            filename = infer_filename(description)
            saved_path = save_query_file(code, filename)
            
            try:
                output = run_query(filename)
            except Exception as run_exc:
                output = f"Execution error: {run_exc}"
                
            try:
                from code_history_db import add_code_entry
                entry_id = add_code_entry(description, code)
            except Exception:
                entry_id = None

            return {
                "code": code,
                "id": entry_id,
                "saved_file": str(saved_path),
                "filename": saved_path.name,
                "crud": True,
                "output": output
            }
        else:
            match = re.search(r"```(\w+)?\n([\s\S]*?)\n```", raw_answer)
            code = f"```{match.group(1) or ''}\n{match.group(2)}\n```" if match else raw_answer
            
            try:
                from code_history_db import add_code_entry
                entry_id = add_code_entry(description, code)
            except Exception:
                entry_id = None
                
            return {
                "code": code,
                "id": entry_id,
                "crud": False,
                "output": "Code generated (non-CRUD query, execution skipped)."
            }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/queries/{filename}/run")
def api_run_query(filename: str):
    try:
        output = run_query(filename)
        return {"status": "success", "output": output}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/login")
def api_login(body: LoginBody, request: Request):
    ok, name = login_student(body.student_id, body.password)
    if not ok:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    request.session["student_id"] = body.student_id
    request.session["name"] = name
    return {"message": f"Login successful. Welcome, {name}!"}


@app.post("/records")
def api_create(body: RecordCreate, request: Request, student_id: int = Depends(_student_id)):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO records (student_id, course, grade) VALUES (?, ?, ?)",
        (student_id, body.course, body.grade),
    )
    conn.commit()
    conn.close()
    save_document(
        f"INSERT INTO records (student_id, course, grade) VALUES ({student_id}, '{body.course}', '{body.grade}')",
        "create_query",
    )
    return {"message": "Record added successfully"}


@app.get("/records")
def api_read(student_id: int = Depends(_student_id)):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, course, grade FROM records WHERE student_id = ? ORDER BY id",
        (student_id,),
    )
    rows = [dict(row) for row in cur.fetchall()]
    conn.close()
    return rows


@app.put("/records/{record_id}")
def api_update(record_id: int, body: RecordUpdate, student_id: int = Depends(_student_id)):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "UPDATE records SET grade = ? WHERE id = ? AND student_id = ?",
        (body.grade, record_id, student_id),
    )
    if cur.rowcount == 0:
        conn.close()
        raise HTTPException(status_code=404, detail="Record not found")
    conn.commit()
    conn.close()
    save_document(
        f"UPDATE records SET grade = '{body.grade}' WHERE id = {record_id} AND student_id = {student_id}",
        "update_query",
    )
    return {"message": f"Record {record_id} updated"}


@app.delete("/records/{record_id}")
def api_delete(record_id: int, student_id: int = Depends(_student_id)):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "DELETE FROM records WHERE id = ? AND student_id = ?",
        (record_id, student_id),
    )
    if cur.rowcount == 0:
        conn.close()
        raise HTTPException(status_code=404, detail="Record not found")
    conn.commit()
    conn.close()
    save_document(
        f"DELETE FROM records WHERE id = {record_id} AND student_id = {student_id}",
        "delete_query",
    )
    return {"message": f"Record {record_id} deleted"}
