# Checkpoint transfer diagnosis and bounded local repair

The previous goal turn was an execution failure that changed the next action: authenticate and stage the compatible comparator before spending GPU time. It did not yield a teacher/student research result. This turn completed a bounded local operational repair; no remote compute was allocated.

## Evidence

A one-MiB range probe to each pinned checkpoint returned exactly the requested range and correct advertised total length. The first-drafter header contained 58 tensors and the xPress header 67. Transfers took 10.97 and 13.84 seconds respectively from this Windows machine. These observations establish current endpoint reachability and range support, not whole-file authentication, sustained speed or Modal throughput. No full checkpoint was downloaded.

The original failed cloud record lacks byte/time counters; its slow-transfer cause remains unknown. We cannot claim this repair fixes network throughput or that a new cloud attempt would finish.

## Local changes

`experiments/checkpoint_transfer.py` uses smaller bounded reads, records received bytes, elapsed time, HTTP status, mean rate and projected remaining time, and preserves terminal failure details. Redirected signed URLs are omitted. Whole-file size and SHA256 remain mandatory; unexpected partial responses fail. Existing files cannot be overwritten. There are no retries or resume behavior.

After at least 30 seconds, a conservative scheduling guard requires 1.5 times the projected remaining transfer time to fit before the deadline and 45-second staging reserve. This extrapolation can reject a transfer that would later accelerate; it is an operational admission rule, not proof of network capacity. The independent hard worker deadline remains in place. Progress is committed remotely at bounded intervals by the staging caller, and on terminal events.

Seven local known-answer checks passed: valid authentication, hash mismatch, truncation, extra bytes, partial response, slow transfer with retained counters, and no overwrite. Source/serialization checks pass for the revised bundle. The original failed attempt's frozen plan and 56 source hashes remain unchanged; the working plan explicitly disables launch and the existing-attempt guard still prohibits resubmission.

## Next decision

Keep the same scientific comparison. Do not fund another identical blind transfer. Before another paid attempt, establish a credible checkpoint-staging route or a separately bounded transfer-readiness step. The documented Hub client supports pinned revisions and local caching, but neither its installation nor faster cloud throughput has been validated here; no switch is assumed to solve the failure. Source: https://huggingface.co/docs/huggingface_hub/guides/download .

The research goal remains compression of useful second-pass corrections. This transfer repair is supporting work, not a research contribution or progress in generation speed. No student fit is justified by it. Funds remain $0 unallocated under the $41 goal authorization, including unsettled allowances. No new spending, no new reservation, no new paid attempt, and no automatic reclaiming of historical holds.
