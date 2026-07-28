# Local bridge for the setup dashboard — runs the same engine as the cloud cron,
# but on this PC, so it needs no GitHub Actions / billing. A Windows Scheduled
# Task runs this every 2 hours:
#   refresh Dukascopy data -> detect -> Telegram-alert new setups -> push
# The push updates the GitHub repo, which redeploys your Vercel dashboard.
#
# Telegram creds come from the gitignored .env at the project root.

$proj = Split-Path -Parent $PSScriptRoot          # scripts/ -> project root
Set-Location $proj
$log = Join-Path $PSScriptRoot "refresh_local.log"
function Log($m) { "$(Get-Date -Format o)  $m" | Out-File -Append -Encoding utf8 $log }

Log "starting"

uv run python scripts/live_scan.py *>> $log
if ($LASTEXITCODE -ne 0) { Log "live_scan failed (exit $LASTEXITCODE) - not pushing"; exit 1 }

git add vercel-deploy/public vercel-deploy/seen.json
git diff --cached --quiet
if ($LASTEXITCODE -ne 0) {
    git commit -q -m "live-scan: local refresh $(Get-Date -Format o)" *>> $log
    git push *>> $log
    Log "pushed refresh"
} else {
    Log "no change this run"
}
