# Task-specific cohort extension proposal

The amended-189 sequence is a governed T0 cohort. It should not be forced to supply every task-specific relation.

Minimal recovery path:

1. Keep amended-189 immutable for T0/T1/T2G.
2. Build a T4L edit cohort directly from locked public T4L rows where `question_a`, `answer_a`, `question_b`, `answer_b`, image, and distinct lesion roles are complete.
3. Run frozen base inference on the task-specific candidate union; retain T4L rows only when question A is base-wrong and question B is base-correct.
4. Use the same approach for T2L, T3L/T3G, and T4G if the operator wants denominators independent of the amended-189 overlap.
5. Freeze cohort selection before any method output is read. Keep stable event keys and one-to-many lineage.
6. Re-run the Core-9 coverage gate; only then qualify methods and authorize formal GPU work.

This proposal does not choose or freeze a new cohort. It changes neither the amended sequence nor any historical raw artifact.

