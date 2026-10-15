# 服务器迁移与复现实验清单

更新日期：2026-10-15

## 本地作为主版本

本地目录同时包含早期人声保护实验和最新真实翻唱路径实验：

- `experiments/nonuniform_noise/`：早期非均匀噪声、人声保护、context gate 等实验。
- `ACE-Step-1.5/experiments/data/covers80_probe/`：真实翻唱路径、三样本汇总、
  Faith 路径辅助上界实验。
- `ACE-Step-1.5/experiments/*.py` 与 `*.sh`：实验、评分和绘图脚本。
- `ACE-Step-1.5/acestep/models/common/vocal_noise_schedule.py`
- `ACE-Step-1.5/acestep/models/common/vocal_noise_schedule_test.py`
- `ACE-Step-1.5/acestep/models/turbo/modeling_acestep_v15_turbo.py`

迁移时不删除服务器旧目录，先将其改名为带日期的备份；新目录校验完成后再决定是否删除备份。

## 不需要传输

- `ACE-Step-1.5/checkpoints/`
- `.venv/`、`__pycache__/`、缓存和日志缓存
- `ACE-Step-1.5/experiments/data/covers80.tgz`
- `ACE-Step-1.5/experiments/data/coversongs/`
- `experiments/eval_models/`

## 生成模型

主模型仓库：`ACE-Step/Ace-Step1.5`

当前实验实际使用：

- `acestep-v15-turbo`
- `vae`
- `Qwen3-Embedding-0.6B`

当前 cover/path 实验关闭了 LLM thinking，未依赖 `acestep-5Hz-lm-1.7B`；如果需要运行
完整 ACE-Step 功能，可将它一并下载。

项目自带下载命令：

```bash
uv run acestep-download
```

## 评测模型

- 风格与音频表征：`laion/clap-htsat-unfused`
- 音乐表征：`m-a-p/MERT-v1-95M`
- 歌词识别：`Qwen/Qwen3-ASR-1.7B`

建议放置到：

```text
experiments/eval_models/clap-htsat-unfused
experiments/eval_models/MERT-v1-95M
experiments/eval_models/Qwen3-ASR-1.7B
```

## 数据集

- 翻唱路径实验：Covers80
- 官方地址：`https://www.labrosa.org/projects/coversongs/covers80/`

原始数据可在服务器重新下载；需要迁移的只是已经裁剪、对齐并产生结果的
`covers80_probe/`。

## 最新关键结果

- 三样本汇总：`ACE-Step-1.5/experiments/data/covers80_probe/summary.md`
- 三样本图：`ACE-Step-1.5/experiments/data/covers80_probe/summary.png`
- Faith 路径辅助报告：
  `ACE-Step-1.5/experiments/data/covers80_probe/faith/oracle_path_report.md`
- Faith 路径辅助图：
  `ACE-Step-1.5/experiments/data/covers80_probe/faith/oracle_path_result.png`

## 迁移后校验

1. 对比文件数量和总大小。
2. 对关键 JSON、PNG、WAV 和 `trajectory_tensors.pt` 计算 SHA-256。
3. 运行实验脚本的 `py_compile` 和路径分解单元测试。
4. 确认服务器旧目录备份仍可恢复。
