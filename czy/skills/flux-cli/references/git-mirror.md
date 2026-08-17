# 训练版本的 GitHub/GitLab 镜像推送

> 强制顺序：GitHub 推送并核对成功 → 创建并启动训练 → 确认任务运行 → GitLab 镜像推送。

## 固定配置

- GitHub：使用当前仓库的 `origin`，是 RL 训练任务唯一允许的代码源；训练任务的 `codeUrl` 必须指向 GitHub。
- GitLab：使用当前工作区已配置的 `gitlab` 远程，仅作为训练启动后的版本镜像；不得把 GitLab 地址填写到训练任务的 `codeUrl`。
- GitLab 远程建议命名为 `gitlab`；凭据通过 Git Credential Manager 或系统凭据存储提供，不要把 Token 写入远程 URL、脚本、JSON 或实验记录。

## 推送时机与去重

1. 新训练版本必须先推送 GitHub，并用 `git ls-remote --heads origin <branch>` 核对远端与训练 commit SHA 完全一致。
2. GitHub 推送失败、远端 SHA 不一致、发生非 fast-forward 或需要强制推送时，停止流程；不得创建或启动训练任务，也不得继续推送 GitLab。
3. 创建 RL 任务时，`taskCodeInfo.codeUrl` 必须使用 GitHub 地址和对应分支；不得使用 GitLab 地址。先 `flux task create --dry-run`，再正式创建。
4. 正式执行 `flux task run` 后，必须通过 `flux task info` 确认任务状态为运行中（状态码 `3`），确认前不推送 GitLab。
5. 只有任务已进入运行态后，才将同一分支、同一 commit 推送到 GitLab。仅创建草稿任务、代码修改但尚未训练、查看状态、监控、下载产物或重复检查时，不推送 GitLab。
6. 同一训练版本只推送一次。以训练任务实际使用的 commit SHA 作为版本标识；若 GitLab 目标分支已经指向该 SHA，直接跳过推送。
7. 恢复训练只有在使用新的代码 commit、或用户明确把它作为新的代码版本时才推送；仅更换 checkpoint 不重复推送。

## 新训练版本启动前的流程

1. 查看工作区状态并确认训练要使用的分支和 commit：

   ```powershell
   git status --short
   git branch --show-current
   git rev-parse HEAD
   ```

2. 确认 GitHub `origin` 远程存在，并确认 GitLab `gitlab` 远程已配置；不存在时添加，不要重复添加：

   ```powershell
   git remote get-url origin
   git remote get-url gitlab
   ```

   如果 `gitlab` 不存在，先根据用户确认的镜像地址添加；不要擅自猜测地址。

3. 在创建/启动训练任务前，将同一分支的同一 commit 推送到 GitHub：

   ```powershell
   git push origin <branch>:<branch>
   git ls-remote --heads origin <branch>
   ```

4. 使用 GitHub 地址创建 RL 任务，正式运行后确认状态为 `3`：

   ```powershell
   flux --yes task create --file ./create-train.json --dry-run
   flux --yes task create --file ./create-train.json
   flux --yes task run --task-id "<task_id>"
   flux --yes task info --task-id "<task_id>"
   ```

   只有 `task info` 显示运行状态 `3` 后，才能继续下一步。

5. 任务已启动后，将同一分支的同一 commit 推送到 GitLab：

   ```powershell
   git push gitlab <branch>:<branch>
   git ls-remote --heads gitlab <branch>
   ```

6. 用两个远端的 `git ls-remote` 结果核对 commit SHA。若远端不是预期 commit、出现非 fast-forward 或需要强制推送，停止并请求用户确认；禁止默认 `--force`。在实验记录中记录训练任务、分支和 commit SHA，但不要记录 Token 或完整凭据。

## 安全边界

- 不打印 GitLab Token、完整凭据、凭据管理器内容或包含凭据的远程 URL。
- 不把 GitLab 镜像推送放进 15 分钟监控循环、回放流程或每次 GitHub 推送钩子。
- GitHub 推送失败或训练未进入运行态时，不得推送 GitLab。
- GitLab 推送失败时，不要自动停止已经运行的训练任务；先报告失败原因并确认远端状态。
