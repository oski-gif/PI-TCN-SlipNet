#!/bin/bash
# Restarts the longer-rollout retrain after its first launch ran out of Windows commit
# memory at startup (0 training steps, no policy). Same script and config, still 4 envs.
# Keeps the failed launch's logs, then waits for enough free commit: 5 processes at
# about 3 GB each plus a margin. Rules: results.md, "Pre-registered: longer rollouts".
DATA="/c/Users/oskic/PI-TCN-SlipNet/Data"
KEEP="$DATA/nsteps512_failed_startup_2026-11-03"
NEED_GB=17

free_gb() {
  powershell.exe -NoProfile -Command \
    "[int][math]::Floor((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1MB)" | tr -d '\r'
}

until grep -q RETRAIN_DONE "$DATA/_run_nsteps_retrain.out"; do sleep 10; done
sleep 10
mkdir -p "$KEEP"
mv "$DATA"/_nsteps512_seed*_attempt_*.log "$KEEP"/
cp "$DATA/_run_nsteps_retrain.out" "$KEEP"/
echo "$(date): failed launch's logs saved to $KEEP"

until [ "$(free_gb)" -ge "$NEED_GB" ]; do
  echo "$(date): $(free_gb) GB commit free, need $NEED_GB, waiting"
  sleep 60
done

echo "$(date): $(free_gb) GB commit free, starting retrain"
bash "$DATA/_run_nsteps_retrain.sh" > "$DATA/_run_nsteps_retrain.out" 2>&1
echo RESTART_DONE
