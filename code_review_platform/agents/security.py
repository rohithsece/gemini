import ast
import re
from typing import Dict, Any, List
from .base import BaseGeminiAgent

SYSTEM_INSTRUCTION = """You are an expert Python code reviewer specializing in software security and vulnerability assessment.
Your goal is to inspect the code for security vulnerabilities, including OWASP Top 10 (SQL Injection, XSS, Command Injection/RCE, Path Traversal), hardcoded credentials/secrets/tokens, insecure cryptographic algorithms (e.g. MD5, SHA1), unsafe deserialization, temporary file misuse, and exposure of sensitive data.

You must respond with valid JSON containing:
- 'pass': boolean (true if there are no high or medium severity vulnerabilities, false otherwise)
- 'summary': string (brief summary of security findings)
- 'findings': array of finding objects, where each object has:
  - 'type': string ('Secret Leak', 'SQL Injection', 'Command Injection', 'Insecure Cryptography', 'Remote Code Execution', 'XSS', 'Path Traversal', 'Other Security Issue')
  - 'message': string (clear explanation of the security risk)
  - 'severity': string ('low', 'medium', 'high')
  - 'line_number': integer or null (specific line of code)
  - 'suggestion': string (concrete guidelines on how to remediate)
"""

SCHEMA_DESCRIPTION = "JSON object with keys: pass (bool), summary (str), findings (list of objects with: type, message, severity, line_number, suggestion)"

class SecurityAgent(BaseGeminiAgent):
    def __init__(self):
        super().__init__("SecurityReview", SYSTEM_INSTRUCTION)

    async def analyze(self, code: str) -> Dict[str, Any]:
        """Runs Gemini-powered security analysis. Falls back to local AST and regex checks if API is unavailable."""
        prompt = f"Analyze the following Python code for security issues and vulnerabilities:\n\n```python\n{code}\n```"
        try:
            result = await self.call_gemini(prompt, SCHEMA_DESCRIPTION)
            result["source"] = "Gemini API"
            return result
        except Exception as e:
            return self.run_fallback(code, error_msg=str(e))

    def run_fallback(self, code: str, error_msg: str) -> Dict[str, Any]:
        """Local static security analysis using AST parsing and regex heuristics."""
        findings = []
        is_pass = True
        
        # Regex checks for secrets (run line-by-line)
        lines = code.splitlines()
        secret_pattern = re.compile(
            r"(?:key|password|secret|token|passwd|credential|private_key|auth_token)\s*=\s*['\"]([^'\"]{8,})['\"]",
            re.IGNORECASE
        )
        # Exclude common dummy placeholders to reduce false positives in test files
        placeholders = {"your_api_key", "placeholder", "secret_here", "dummy", "test", "mysecret", "mypassword", "your_key"}
        
        for idx, line in enumerate(lines, 1):
            stripped = line.strip()
            
            # 1. Regex check for hardcoded secrets
            match = secret_pattern.search(stripped)
            if match:
                secret_value = match.group(1).lower()
                # Check if it is not just a placeholder
                if not any(ph in secret_value for ph in placeholders):
                    findings.append({
                        "type": "Secret Leak",
                        "message": "Potential hardcoded secret or API key assignment detected.",
                        "severity": "high",
                        "line_number": idx,
                        "suggestion": "Remove hardcoded credentials. Load secrets from environment variables or a secure vault."
                    })
                    is_pass = False

        # 2. AST parsing for structural security issues
        try:
            tree = ast.parse(code)
            
            class SecurityVisitor(ast.NodeVisitor):
                def __init__(self):
                    self.findings = []
                    self.is_pass = True

                def visit_Call(self, node: ast.Call):
                    # Check for eval() and exec()
                    if isinstance(node.func, ast.Name):
                        if node.func.id in ("eval", "exec"):
                            self.findings.append({
                                "type": "Remote Code Execution",
                                "message": f"Use of unsafe '{node.func.id}' built-in function.",
                                "severity": "high",
                                "line_number": node.lineno,
                                "suggestion": "Avoid using eval() or exec() with untrusted inputs, as it can lead to arbitrary code execution. Refactor to use safe alternatives like json.loads() or dictionary mappings."
                            })
                            self.is_pass = False
                            
                        # Check for tempfile.mktemp
                        elif node.func.id == "mktemp":
                            self.findings.append({
                                "type": "Other Security Issue",
                                "message": "Use of insecure 'tempfile.mktemp' function.",
                                "severity": "medium",
                                "line_number": node.lineno,
                                "suggestion": "tempfile.mktemp is deprecated and insecure because the file name could be hijacked. Use tempfile.mkstemp or tempfile.TemporaryFile instead."
                            })
                            self.is_pass = False

                    # Check for subprocess / os.system call issues
                    elif isinstance(node.func, ast.Attribute):
                        # os.system(...)
                        if isinstance(node.func.value, ast.Name) and node.func.value.id == "os" and node.func.attr == "system":
                            self.findings.append({
                                "type": "Command Injection",
                                "message": "Use of unsafe 'os.system' for running shell commands.",
                                "severity": "high",
                                "line_number": node.lineno,
                                "suggestion": "Avoid os.system as it passes inputs directly to the shell. Use subprocess.run with shell=False and pass arguments as a list."
                            })
                            self.is_pass = False
                        
                        # subprocess calls with shell=True
                        elif isinstance(node.func.value, ast.Name) and node.func.value.id == "subprocess":
                            # Check keywords for shell=True
                            has_shell_true = False
                            for kw in node.keywords:
                                if kw.arg == "shell" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                                    has_shell_true = True
                            
                            if has_shell_true:
                                self.findings.append({
                                    "type": "Command Injection",
                                    "message": f"subprocess.{node.func.attr} invoked with shell=True.",
                                    "severity": "high",
                                    "line_number": node.lineno,
                                    "suggestion": "Set shell=False and pass command and arguments as a list of strings to prevent command injection."
                                })
                                self.is_pass = False

                        # hashlib insecure crypt check
                        elif isinstance(node.func.value, ast.Name) and node.func.value.id in ("hashlib", "crypt"):
                            if node.func.attr in ("md5", "sha1"):
                                self.findings.append({
                                    "type": "Insecure Cryptography",
                                    "message": f"Use of weak hashing algorithm '{node.func.attr}'.",
                                    "severity": "medium",
                                    "line_number": node.lineno,
                                    "suggestion": "MD5 and SHA-1 are cryptographically broken and vulnerable to collision attacks. Use SHA-256 (hashlib.sha256) or stronger algorithms instead."
                                })
                                self.is_pass = False

                    self.generic_visit(node)

                # Check for SQL injection in execute methods
                def visit_Attribute(self, node: ast.Attribute):
                    # Check for cursor.execute or conn.execute
                    if node.attr == "execute":
                        # We need to see if the first argument of the call is a formatted string or string concatenation
                        parent = self.get_call_parent(node)
                        if parent and len(parent.args) > 0:
                            sql_arg = parent.args[0]
                            is_insecure = False
                            # F-string e.g. f"SELECT * FROM user WHERE id = {user_id}"
                            if isinstance(sql_arg, ast.JoinedStr):
                                is_insecure = True
                            # String concatenation e.g. "SELECT * FROM user WHERE id = " + user_id
                            elif isinstance(sql_arg, ast.BinOp) and isinstance(sql_arg.op, ast.Add):
                                is_insecure = True
                            # String formatting e.g. "SELECT * FROM user WHERE id = %s" % user_id
                            elif isinstance(sql_arg, ast.BinOp) and isinstance(sql_arg.op, ast.Mod):
                                is_insecure = True

                            if is_insecure:
                                self.findings.append({
                                    "type": "SQL Injection",
                                    "message": "Dynamic SQL query construction detected in database execute call.",
                                    "severity": "high",
                                    "line_number": node.lineno,
                                    "suggestion": "Always use parameterized queries (prepared statements) to pass variables instead of dynamic string formatting or concatenation."
                                })
                                self.is_pass = False
                    self.generic_visit(node)

                def get_call_parent(self, node):
                    # Traverse upwards or match parent call node structure if possible.
                    # Since AST doesn't have parent links by default, we just check if this is the attribute function of a call
                    # which is typical for visits. We can verify if the node is within call context.
                    return None

            # We can override get_call_parent by tracking node contexts, or keep it simple.
            # To be robust, let's write a simple custom parent-tracking or search in parent contexts.
            visitor = SecurityVisitor()
            # To track parents, we can inject parent attributes or use a custom node walker.
            # Let's write a robust walker that checks database calls
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "execute":
                    # It's cursor.execute(...)
                    if len(node.args) > 0:
                        sql_arg = node.args[0]
                        is_insecure = False
                        # f"..."
                        if isinstance(sql_arg, ast.JoinedStr):
                            is_insecure = True
                        # binop addition or modulo
                        elif isinstance(sql_arg, ast.BinOp):
                            if isinstance(sql_arg.op, (ast.Add, ast.Mod)):
                                is_insecure = True
                            # Wait, check if there is string concatenation nested
                        elif isinstance(sql_arg, ast.Call) and isinstance(sql_arg.func, ast.Attribute) and sql_arg.func.attr == "format":
                            # "SELECT ... {}".format(val)
                            is_insecure = True
                        
                        if is_insecure:
                            visitor.findings.append({
                                "type": "SQL Injection",
                                "message": "Insecure string formatting/interpolation in database execute statement.",
                                "severity": "high",
                                "line_number": node.lineno,
                                "suggestion": "Utilize query parameters (e.g. ?, %s, :val depending on the driver) instead of string formatting or concatenation."
                            })
                            visitor.is_pass = False

            visitor.visit(tree)
            findings.extend(visitor.findings)
            is_pass = is_pass and visitor.is_pass

        except SyntaxError:
            # Fallback to simple regex for SQL injection if AST compile fails
            sql_inj_pattern = re.compile(
                r"\.execute\(\s*(f['\"]|['\"].*?%|['\"].*?\{\}.*?\.format)",
                re.IGNORECASE
            )
            for idx, line in enumerate(lines, 1):
                if sql_inj_pattern.search(line):
                    findings.append({
                        "type": "SQL Injection",
                        "message": "Potential SQL injection pattern in query execution.",
                        "severity": "high",
                        "line_number": idx,
                        "suggestion": "Ensure SQL queries use proper parameter bindings rather than string interpolation."
                    })
                    is_pass = False
        except Exception as exc:
            pass

        summary = (
            "No high or medium security issues detected (Local Fallback)" if is_pass
            else f"Detected {len(findings)} security issue(s) using local static analysis."
        )

        return {
            "pass": is_pass,
            "summary": summary,
            "findings": findings,
            "source": f"Local Fallback (Gemini Error: {error_msg[:100]}...)"
        }
