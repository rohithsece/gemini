import re
from typing import Dict, Any
from .base import BaseGeminiAgent

SYSTEM_INSTRUCTION = """You are an expert Python code reviewer specializing in code quality and style analysis.
Your goal is to inspect code for naming conventions (snake_case, PascalCase, UPPER_CASE), PEP 8 adherence, readability, formatting issues, cyclomatic/cognitive complexity, docstrings, variable definitions, and maintainability.

You must respond with valid JSON containing:
- 'pass': boolean (true if overall score >= 7.0, false otherwise)
- 'score': float (rating of code quality from 0.0 to 10.0)
- 'summary': string (brief summary of style & layout evaluation)
- 'findings': array of finding objects, where each object has:
  - 'type': string ('Naming Convention', 'PEP 8 Style', 'Complexity', 'Readability', 'Documentation', 'Typing')
  - 'message': string (clear explanation of the quality issue)
  - 'severity': string ('low', 'medium', 'high')
  - 'line_number': integer or null (specific line of code)
  - 'suggestion': string (concrete suggestion to refactor or format)
"""

SCHEMA_DESCRIPTION = "JSON object with keys: pass (bool), score (float), summary (str), findings (list of objects with: type, message, severity, line_number, suggestion)"

class CodeQualityAgent(BaseGeminiAgent):
    def __init__(self):
        super().__init__("CodeQuality", SYSTEM_INSTRUCTION)

    async def analyze(self, code: str) -> Dict[str, Any]:
        """Runs Gemini-powered code quality analysis. Falls back to local metrics if API is unavailable."""
        prompt = f"Review code quality, naming, PEP 8, and complexity:\n```python\n{code}\n```"
        try:
            result = await self.call_gemini(prompt, SCHEMA_DESCRIPTION)
            result["source"] = "Gemini API"
            return result
        except Exception as e:
            return self.run_fallback(code, error_msg=str(e))

    def run_fallback(self, code: str, error_msg: str) -> Dict[str, Any]:
        """Local static style evaluation based on regex heuristics."""
        findings = []
        base_score = 10.0
        lines = code.splitlines()
        
        # Heuristic 1: Function and Class Docstring checks
        # Count classes and functions to verify they have docstring definitions
        defs_without_docs = 0
        total_defs = 0
        
        in_docstring = False
        last_def_line = -1
        
        for idx, line in enumerate(lines, 1):
            stripped = line.strip()
            
            # Line length check (PEP 8 recommends 79-120 chars)
            if len(line) > 100:
                findings.append({
                    "type": "PEP 8 Style",
                    "message": f"Line exceeds 100 characters ({len(line)} chars).",
                    "severity": "low",
                    "line_number": idx,
                    "suggestion": "Split the line into multiple lines or refactor expressions."
                })
                base_score -= 0.2
                
            if stripped.startswith("def ") or stripped.startswith("class "):
                total_defs += 1
                # Check if the next non-empty line starts with triple quotes
                next_idx = idx
                has_doc = False
                while next_idx < len(lines):
                    next_line = lines[next_idx].strip()
                    if next_line:
                        if next_line.startswith('"""') or next_line.startswith("'''"):
                            has_doc = True
                        break
                    next_idx += 1
                if not has_doc:
                    defs_without_docs += 1
                    func_name = stripped.split("(")[0].replace("def ", "").replace("class ", "").strip()
                    findings.append({
                        "type": "Documentation",
                        "message": f"Missing docstring for '{func_name}'.",
                        "severity": "low",
                        "line_number": idx,
                        "suggestion": "Add a informative triple-quoted docstring explaining usage and parameters."
                    })
                    base_score -= 0.5

            # Complexity check: check for nested indent levels
            indent = len(line) - len(line.lstrip())
            if indent > 12: # Greater than 3 tabs/12 spaces indentation usually means deep nested loops/ifs
                findings.append({
                    "type": "Complexity",
                    "message": "High code nesting level detected.",
                    "severity": "medium",
                    "line_number": idx,
                    "suggestion": "Extract deeply nested logic into a helper function."
                })
                base_score -= 0.5

        # Cap score between 0 and 10
        score = max(0.0, min(10.0, base_score))
        passed = score >= 7.0
        
        summary = (
            f"Code quality is good. Score: {score:.1f}/10.0 (Local Fallback)" if passed
            else f"Needs refactoring. Quality score: {score:.1f}/10.0 (Local Fallback)"
        )

        return {
            "pass": passed,
            "score": round(score, 1),
            "summary": summary,
            "findings": findings,
            "source": f"Local Fallback (Gemini Error: {error_msg[:100]}...)"
        }
