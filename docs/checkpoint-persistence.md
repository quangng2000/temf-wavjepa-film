# Persistent checkpoints on object-backed mounts

Use `src.callbacks.durable.DurableCheckpoint` instead of the ordinary checkpoint
callback when the working directory is temporary and backups live on a Global
volume. The normal ModelCheckpoint options still apply. For this production run:

```yaml
_target_: src.callbacks.durable.DurableCheckpoint
dirpath: /opt/temf-workspace/production/RUN_ID/checkpoints
backup_dir: /workspace/temf-production/RUN_ID
run_id: RUN_ID
every_n_train_steps: 1000
monitor: step
mode: max
save_top_k: 2
save_last: link
retention:
  latest: 2
  every: 20000
  prune: true
```

The callback first lets Lightning finish its atomic local checkpoint write.
It then copies that completed file to a unique persistent filename, computes
SHA-256, flushes and fsyncs it, and reads the complete persistent file back to
check its size and hash. A JSON completion receipt is written, fsynced and read
back only after the checkpoint passes. No rename, symlink or lock on the Global
mount is required. The local `last.ckpt` link remains on the local filesystem.

Training waits for backup completion at each checkpoint. Any save, copy,
verification or cleanup failure propagates and stops the training process;
the callback never stops, restarts or releases a pod. A failed copy has no valid
completion receipt and must not be selected for recovery. Its partial file is
left intact for inspection.

Persistent cleanup is disabled unless `retention.prune` is explicitly true.
When enabled, it runs only after a new completed receipt, retaining the latest
two distinct steps and one copy at each 20,000-update milestone. It only removes
files with valid receipts belonging to this exact `run_id` and backup directory.
Unknown files, other runs, and the original source volume remain untouched.
Local retention is separately managed by `save_top_k: 2`.

For recovery, select a completed receipt, copy its named checkpoint to local
storage, verify its recorded byte count and SHA-256, and pass the local file as
`ckpt_path`. Keep the same model/data recipe, `run_id` and backup destination.
Do not load an untrusted checkpoint or treat an unreceipted file as complete.

The receipt labels its check as **fsync plus mounted readback**. This confirms
what the mount reports and returns; it is not an independent query of the remote
storage backend or proof against provider failure. The old source volume remains
retained separately. Backup time is additional to the measured optimizer-update
time, and depends on the mount's throughput.
