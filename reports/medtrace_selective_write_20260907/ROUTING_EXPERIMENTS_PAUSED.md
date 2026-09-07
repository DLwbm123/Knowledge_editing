# Superseded unstarted routing work

Per the selective-write plan, unstarted G0/G1/G2 pooling and D00-D11 router work are paused, not selected or executed by this attempt. Their source, configurations and NOT_RUN reports remain unchanged.

The pooling run `20260907T060435Z_cp_independent_pooling` was checked: 21 PENDING tasks, all attempts=0, no CAMPAIGN_START or PIDS file. A new STOP marker prevents its old worker loop from advancing. No historical report, result, checkpoint or task queue was overwritten. No existing process was terminated.

This selective-write queue does not include D00-D11 or any new router training. No additional D00-D11 running process was identified or stopped in this execution.
