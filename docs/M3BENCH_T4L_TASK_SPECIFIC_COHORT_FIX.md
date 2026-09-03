# M3Bench T4L task-specific cohort correction

The previous T4L hard stop came from treating the governed T0 amended-189
sequence as the only legal edit-anchor pool. It did not show that T4L metadata
was absent.

The public T4L table defines each direction directly: `question_a/answer_a` is
the edit target and `question_b/answer_b` is the same-image locality probe.
The builder therefore locks and audits the 257-row public table, creates stable
query IDs, and freezes only candidates where A is base-wrong and B is
base-correct. Amended-189 membership is never consulted.

This is a public-release-aligned task-specific cohort, not a paper-exact claim.
Method output is not an input to selection.
