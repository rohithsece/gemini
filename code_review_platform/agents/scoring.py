import json
from typing import Dict, Any, List
from .base import BaseGeminiAgent

SYSTEM_INSTRUCTION = """You are an expert software architect and project scorer.
Your goal is to digest the reports from the Bug Detection Agent, the Code Quality Agent, and the Security Review Agent, and compute an overall project score and grade.

You must respond with valid JSON containing:
- 'score': integer (overall project score from 0 to 100, where 100 is perfect and 0 is completely broken/insecure)
- 'grade': string (overall grade: 'A+', 'A', 'B', 'C', 'D', or 'F')
- 'summary': string (high-level executive summary of code quality, security posture, and runtime correctness)
- 'strengths': array of strings (list of specific architectural or programming strengths observed in the code)
- 'recommendations': array of strings (prioritized, actionable steps to improve the code score, starting with the most critical)
"""

SCHEMA_DESCRIPTION = "JSON object with keys: score (int), grade (str), summary (str), strengths (list of strings), recommendations (list of strings)"

class ScoringAgent(BaseGeminiAgent):
    def __init__(self):
        super().__init__("ProjectScoring", SYSTEM_INSTRUCTION)

    async def analyze(
        self,
        bug_report: Dict[str, Any],
        quality_report: Dict[str, Any],
        security_report: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Uses fast deterministic scoring — avoids a 4th Gemini round-trip for minimal latency."""
        return self.run_fallback(bug_report, quality_report, security_report, error_msg="")

    def run_fallback(
        self,
        bug_report: Dict[str, Any],
        quality_report: Dict[str, Any],
        security_report: Dict[str, Any],
        error_msg: str
    ) -> Dict[str, Any]:
        """Deterministic fallback scoring if the Gemini API call fails."""
        base_score = 100.0
        strengths = []
        recommendations = []
        
        # 1. Bug Deductions
        bug_findings = bug_report.get("findings", [])
        has_critical_bugs = False
        for f in bug_findings:
            sev = f.get("severity", "low").lower()
            if sev == "high":
                base_score -= 15
                has_critical_bugs = True
            elif sev == "medium":
                base_score -= 8
            else:
                base_score -= 3
            recommendations.append(f"Bug [{sev.upper()}]: {f.get('message')} (Line {f.get('line_number') or 'N/A'}). Suggestion: {f.get('suggestion')}")

        # 2. Quality Deductions
        # If quality report has a score, we can use it to lower our score
        if "score" in quality_report:
            q_score = quality_report["score"]
            # A quality score of 10 means 0 deduction; score of 0 means 20 points deduction
            base_score -= (10.0 - q_score) * 2.5
        
        quality_findings = quality_report.get("findings", [])
        for f in quality_findings:
            sev = f.get("severity", "low").lower()
            # We don't deduct heavily for quality if we already deducted based on score, but we add recommendations
            recommendations.append(f"Quality [{sev.upper()}]: {f.get('message')} (Line {f.get('line_number') or 'N/A'}). Suggestion: {f.get('suggestion')}")

        # 3. Security Deductions
        sec_findings = security_report.get("findings", [])
        has_critical_sec = False
        for f in sec_findings:
            sev = f.get("severity", "low").lower()
            if sev == "high":
                base_score -= 20
                has_critical_sec = True
            elif sev == "medium":
                base_score -= 10
            else:
                base_score -= 4
            recommendations.append(f"Security [{sev.upper()}]: {f.get('message')} (Line {f.get('line_number') or 'N/A'}). Suggestion: {f.get('suggestion')}")

        # Clamp score between 0 and 100
        score = max(0, min(100, int(round(base_score))))
        
        # Determine Grade
        if score >= 95:
            grade = "A+"
        elif score >= 90:
            grade = "A"
        elif score >= 80:
            grade = "B"
        elif score >= 70:
            grade = "C"
        elif score >= 60:
            grade = "D"
        else:
            grade = "F"

        # Determine Strengths
        if not has_critical_bugs and bug_report.get("pass", True):
            strengths.append("Code is structurally valid and executes without high-severity bugs.")
        if quality_report.get("pass", True) or quality_report.get("score", 10) >= 7.5:
            strengths.append("Demonstrates solid adherence to code readability standards and low cognitive complexity.")
        if not has_critical_sec and security_report.get("pass", True):
            strengths.append("Code displays good baseline security posture with no critical security leaks or execution hazards.")
            
        if not strengths:
            strengths.append("Core logic parses correctly.")

        # Build Summary
        summary = (
            f"Project Review Complete. Calculated score is {score}/100 (Grade {grade}). "
            f"Analyzed {len(bug_findings)} bug issues, {len(quality_findings)} quality indicators, and {len(sec_findings)} security findings."
        )

        return {
            "score": score,
            "grade": grade,
            "summary": summary,
            "strengths": strengths,
            "recommendations": recommendations[:6],  # limit recommendations to top 6 to prevent clutter
            "source": f"Local Fallback (Gemini Error: {error_msg[:100]}...)"
        }
