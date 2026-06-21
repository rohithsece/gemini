import ast
import traceback
from typing import Dict, Any
from .base import BaseGeminiAgent

SYSTEM_INSTRUCTION = """You are an expert Python code reviewer specializing in bug detection.
Your goal is to analyze Python code for syntax errors, logical errors, infinite loops, runtime exceptions, dead code, null/undefined reference errors, and unhandled edge cases.
Be critical, concise, and accurate. Do not report stylistic details (PEP 8) or security vulnerabilities; focus solely on execution correctness and stability.

You must respond with valid JSON containing:
- 'pass': boolean (true if code has no medium or high bugs, false otherwise)
- 'summary': string (brief summary of findings)
- 'findings': array of finding objects, where each object has:
  - 'type': string ('Syntax Error', 'Logical Bug', 'Runtime Exception', 'Edge Case', 'Loop Issue')
  - 'message': string (clear explanation of the issue)
  - 'severity': string ('low', 'medium', 'high')
  - 'line_number': integer or null (specific line of code)
  - 'suggestion': string (concrete instructions on how to fix)
"""

SCHEMA_DESCRIPTION = "JSON object with keys: pass (bool), summary (str), findings (list of objects with: type, message, severity, line_number, suggestion)"

class BugDetectionAgent(BaseGeminiAgent):
    def __init__(self):
        super().__init__("BugDetection", SYSTEM_INSTRUCTION)

    async def analyze(self, code: str) -> Dict[str, Any]:
        """Runs Gemini-powered bug detection. Falls back to local AST parser if API is unavailable."""
        prompt = f"Find execution bugs in this Python code:\n```python\n{code}\n```"
        try:
            result = await self.call_gemini(prompt, SCHEMA_DESCRIPTION)
            # Add a flag to indicate it ran via Gemini
            result["source"] = "Gemini API"
            return result
        except Exception as e:
            # Fallback to local heuristic checks
            return self.run_fallback(code, error_msg=str(e))

    def run_fallback(self, code: str, error_msg: str) -> Dict[str, Any]:
        """Local static analysis using python's built-in AST library."""
        findings = []
        is_pass = True
        
        # 1. Check for Syntax Errors using compile/ast
        try:
            ast.parse(code)
        except SyntaxError as se:
            is_pass = False
            findings.append({
                "type": "Syntax Error",
                "message": f"Syntax error: {se.msg}",
                "severity": "high",
                "line_number": se.lineno,
                "suggestion": "Correct the syntax highlighting or formatting error."
            })
        except Exception as e:
            is_pass = False
            findings.append({
                "type": "Syntax Error",
                "message": f"Compilation failed: {str(e)}",
                "severity": "high",
                "line_number": None,
                "suggestion": "Verify your code is valid Python syntax."
            })

        # 2. Local heuristics (searching for obvious mistakes)
        lines = code.splitlines()
        for idx, line in enumerate(lines, 1):
            stripped = line.strip()
            # Simple division by zero check
            if "/ 0" in stripped or "/0" in stripped:
                findings.append({
                    "type": "Runtime Exception",
                    "message": "Potential division by zero.",
                    "severity": "high",
                    "line_number": idx,
                    "suggestion": "Ensure the denominator is validated to be non-zero before division."
                })
                is_pass = False
            # Check for mutable default arguments
            if "def " in stripped and ("=[]" in stripped or "={}" in stripped):
                findings.append({
                    "type": "Logical Bug",
                    "message": "Mutable default argument used.",
                    "severity": "medium",
                    "line_number": idx,
                    "suggestion": "Use None as the default value and initialize the mutable object inside the function body."
                })
                is_pass = False

        summary = (
            f"No major bugs detected (Local Fallback)" if is_pass
            else f"Detected {len(findings)} bug(s) using local static analysis."
        )

        return {
            "pass": is_pass,
            "summary": summary,
            "findings": findings,
            "source": f"Local Fallback (Gemini Error: {error_msg[:100]}...)"
        }
