# YuE2 Cover（曲风编辑）测试报告

日期：2026-10-13
用途：与 ACE-Step 1.5 Cover 测试同口径的横向评测。

---

## 1. 测试对象与方法

- 模型：YuE2-3B + YuE2-Vae（`m-a-p/YuE2-3B`，48kHz 立体声）
- Cover 流程：源音频 → SheetSage2 转写旋律 ABC（melody-only 去和弦）→ YuE2 `cot="melody"` 重生成
- 样本：与 ACE-Step 相同的 3 首源歌（中文女声 120s / 英文女声 80s / 英文男声 120s）
- 曲风：jazz / rock / electronic / folk
- **输入歌词 = 源歌实际唱的歌词**（`03_corrected_lyrics/*.txt`，与 ACE-Step 喂的完全一致）
- 生成参数：`--budget 10 --offload-ar --offline`（GPU 被占满时低显存模式），`seed=42`
- ASR：Qwen3-ASR-1___7B（bf16，`max_new_tokens=2048`）
- 覆盖率：cover ASR vs 输入歌词，中文按字、英文按词，`difflib` 对齐

> 修正记录：第一版中文误用了 `chinese_female_lyrics.txt`（完整版 304 字），
> 但源歌实际唱的是精简版 165 字（`03_corrected_lyrics/chinese_female.txt`），
> 导致中文覆盖率被错误压低。现已用正确歌词重新生成中文 4 个 cover。

## 2. 产物

| 目录 | 内容 |
|---|---|
| `outputs/cover_{jazz,rock,electronic,folk}/` | 中文女声 4 个 cover（精简版歌词） |
| `outputs/english_female_{...}/` | 英文女声 4 个 cover |
| `outputs/english_male_{...}/` | 英文男声 4 个 cover |
| `outputs/05_asr_yue2/` | 12 个 cover 的 Qwen3-ASR 文本 |
| `cover-score-en-female/` `cover-score-en-male/` | 英文两首的旋律 ABC 转写 |

脚本：`asr_yue2_covers.py`（ASR）、`compare_yue2.py`（覆盖率）。

## 3. 结果：歌词覆盖率（cover ASR vs 输入歌词）

| 歌 | jazz | rock | electronic | folk |
|---|---:|---:|---:|---:|
| 中文女声（165 字） | 95.8% | 89.7% | **97.6%** | 95.8% |
| 英文女声（81 词） | 85.2% | 61.7% | 79.0% | 84.0% |
| 英文男声（37 词） | **97.3%** | 91.9% | 83.8% | 89.2% |

## 4. 与 ACE-Step 的对比（同口径：cover ASR vs 源歌实际唱的歌词）

| 歌 | 模型 | jazz | rock | electronic | folk |
|---|---|---|---|---|---|
| 中文女声 | ACE-Step cns=0.6 | 98.8% | 99.4% | 97.6% | 95.2% |
| | **YuE2** | 95.8% | 89.7% | 97.6% | 95.8% |
| 英文女声 | ACE-Step cns=0.6 | 82.7% | 88.9% | 92.6% | 75.3% |
| | **YuE2** | 85.2% | 61.7% | 79.0% | 84.0% |
| 英文男声 | ACE-Step cns=0.6 | 75.7% | 75.7% | 59.5% | 73.0% |
| | **YuE2** | 97.3% | 91.9% | 83.8% | 89.2% |

> 注：ACE-Step 基准是源音频 ASR（含 ~8% 英文误听），YuE2 基准是校正歌词（干净文本），
> 两者"目标文本"的干净程度略有差异，但都是"源歌实际唱的内容"。

## 5. 关键发现

1. **中文、短英文歌，YuE2 都能稳定合成输入歌词**（中文 90~98%，英文男声 84~97%）。
2. **英文女声（美国国歌 81 词）YuE2 波动最大**（61.7%~85.2%），尤其 rock 只有 61.7%——国歌的弱起节奏 + 长句在 rock 编曲下丢词明显。
3. **YuE2 是"重唱"不是"改"**：歌词是纯文本输入，无音素对齐、不保原唱音色；能唱对多少取决于生成时能否把整段歌词 token 吐完，长句/快节奏易丢词。
4. 与 ACE-Step 相比：短歌（英文男声）YuE2 更稳；英文女声长句 YuE2 反而更易丢词；中文两者相当。

## 6. 遗留问题（可作研究点）

1. 英文长句/弱起节奏下的歌词丢词（rock 61.7%），是 YuE2"重唱"路径的短板。
2. YuE2 无音素对齐、无原唱音色保持（官方自曝），cover 只保旋律不保歌手身份。
3. 曲风强度（style caption）对歌词保真的影响：rock 最差、jazz/folk 较好，值得单独做"曲风 vs 歌词保真"的扫档。
