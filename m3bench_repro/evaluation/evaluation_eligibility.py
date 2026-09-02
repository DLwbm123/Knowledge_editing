from __future__ import annotations
def require_evaluation_eligible(record:dict)->None:
 if not record.get("evaluation_eligible",True): raise ValueError("record excluded from evaluation")
def require_task_eligible(record:dict)->None:
 require_evaluation_eligible(record)
 if record.get("is_correct") is None: raise ValueError("null correctness cannot become baseline-wrong")
