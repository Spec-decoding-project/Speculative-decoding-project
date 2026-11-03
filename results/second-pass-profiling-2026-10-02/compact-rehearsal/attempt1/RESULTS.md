# Targeted rehearsal attempt: staging failure, learning untested

The authorized attempt stopped before GPU submission. No optimizer updates or evaluations ran. This is an operational failure, not evidence for or against targeted rehearsal.

## Evidence

Local launch checks passed after disk availability rose to 2.74 GiB; no cleanup was performed. The application uploaded the authenticated saved inputs and submitted one staging call. The recovered preflight record reports torch 2.9.1+cu128 and successful imports. In the frozen staging code this record follows source/data/checkpoint verification and precedes the first explicit volume commit. The later staged.json record is absent. Thus staging reached that boundary, but the exact cause of failure is unresolved; the absence of later records does not prove a storage-service defect.

The controller timed out waiting 325 seconds for staging. Provider logs report repeated runner exit code 124, corresponding to the prepared hard deadline. There was one controller staging submission, with retries=0, and no GPU call ID. Repeated runner exits mean platform restarts may have occurred; do not claim that the platform executed only once. The provider call graph marks stage_work failed with no child calls.

## Closure and cost

Elapsed through closure: 5.67 minutes. App stopped and zero active tasks independently confirmed. One 48-byte preflight artifact recovered and its recorded local hash verified. No remote manifest digest was recovered, so this is not an independent source-to-recovery hash comparison. Owned temporary volume deletion verified by controller. Provider billing rows are not yet available: cost is unknown, not zero. The full $0.95 allowance remains retained pending reconciliation; campaign unallocated balance $0.03506668. No retry launched.

## Next step

Preserve this frozen attempt. Before a further paid attempt, diagnose the staging boundary and add timestamps before and after hashing, imports and commits, with a durable completion record. Correct the recovery-reserve estimate only from evidence; do not modify training, outcomes or launch another fit to debug staging. The experiment remains untested.
