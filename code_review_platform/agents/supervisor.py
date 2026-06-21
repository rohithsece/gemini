import asyncio
from typing import Dict, Any

from .bug_detection import BugDetectionAgent
from .code_quality import CodeQualityAgent
from .security import SecurityAgent
from .scoring import ScoringAgent

class Supervisor:
    def __init__(self):
        self.bug_agent = BugDetectionAgent()
        self.quality_agent = CodeQualityAgent()
        self.security_agent = SecurityAgent()
        self.scoring_agent = ScoringAgent()

    async def run_all(self, code: str, language: str = "python", filename: str = "main.py") -> Dict[str, Any]:
        """Runs Bug, Quality, and Security agents in parallel, then aggregates findings 
        using the Project Scoring agent. Returns the full review dictionary.
        """
        # Run standard checks concurrently
        bug_task = self.bug_agent.analyze(code)
        quality_task = self.quality_agent.analyze(code)
        security_task = self.security_agent.analyze(code)
        
        bug_report, quality_report, security_report = await asyncio.gather(
            bug_task, quality_task, security_task
        )
        
        # Run scoring analysis on the consolidated reports
        scoring_report = await self.scoring_agent.analyze(
            bug_report=bug_report,
            quality_report=quality_report,
            security_report=security_report
        )
        
        return {
            "bug_report": bug_report,
            "quality_report": quality_report,
            "security_report": security_report,
            "scoring_report": scoring_report
        }
