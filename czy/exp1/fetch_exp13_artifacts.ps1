# Download exp1.3 replay artifacts from TASK_20260818_148 (URLs not echoed)
$ErrorActionPreference = "Stop"
$dest = "e:\X1\F1_one\F1_one\czy\data\exp1.3"
New-Item -ItemType Directory -Path $dest -Force | Out-Null
$models = flux task model list --task-id TASK_20260818_148 --page 1 --limit 10 | ConvertFrom-Json
$rows = $models.data.rows
$video = $rows | Where-Object { $_.fileName -eq 'play_adaptive.mp4' } | Select-Object -First 1
$csvpt = $rows | Where-Object { $_.fileName -eq 'model_isaac_csv.pt' } | Select-Object -First 1
$model = $rows | Where-Object { $_.fileName -eq 'model_5000.pt' } | Select-Object -First 1
if ($video -and $video.videoUrlDown) { curl.exe -s --ssl-no-revoke -L -o "$dest\play_output.mp4" $video.videoUrlDown; "mp4: $((Get-Item "$dest\play_output.mp4").Length) bytes" }
if ($csvpt -and $csvpt.policUrlDown) { curl.exe -s --ssl-no-revoke -L -o "$dest\model_isaac_csv.pt" $csvpt.policUrlDown; "csvpt: $((Get-Item "$dest\model_isaac_csv.pt").Length) bytes" }
if ($model -and $model.policUrlDown) { curl.exe -s --ssl-no-revoke -L -o "$dest\model.pt" $model.policUrlDown; "model: $((Get-Item "$dest\model.pt").Length) bytes" }
Remove-Item "$dest\model_isaac_csv.pt" -ErrorAction SilentlyContinue
