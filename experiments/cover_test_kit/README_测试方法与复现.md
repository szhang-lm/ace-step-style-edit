# ACE-Step 1.5 Cover（曲风编辑）测试：方法、样本与结果

日期：2026-10-11（本包整理）
用途：可复现的 Cover 能力评测，供在其他模型（如 YuE2）上做同样评测时对照。

---

## 1. 目的

评测 ACE-Step 1.5 的 **Cover 模式**能否在**改变曲风**的同时**保持歌词完整**。

核心结论（痛点）：

> 曲风变化程度 与 歌词保真 被绑死在**同一个物理量**（"离源音频多近"）上，模型**没有独立的曲风/唱法强度通道**。想曲风变得大 → 歌词必然丢；想歌词保得全 → 曲风接近原唱。

---

## 2. 测试样本（3 首源歌）

音频格式：**48 kHz / 立体声 / 16-bit WAV**。

| 源音频文件 | 时长 | 语言/人声 | 来源 | 歌词内容 |
|---|---|---|---|---|
| `sources/chinese_female_120s.wav` | 120s | 中文女声 | ACE-Step 受控生成 | 原创中文歌词 |
| `sources/english_female_80s.wav` | 80s | 英文女声 | 真实专业录音 | 美国国歌 *The Star-Spangled Banner* |
| `sources/english_male_120s.wav` | 120s | 英文男声 | 真实专业录音 | *America the Beautiful*（第一段 + 间奏） |

- 中文女声歌词见 `sources/chinese_female_lyrics.txt`。
- 三首歌的**校正歌词**（实际喂给模型的文本）见 `03_corrected_lyrics/*.txt`。

---

## 3. 曲风（4 种）与 style caption

| 曲风 | style caption（英文） |
|---|---|
| jazz | `smooth jazz, upright piano, brushed drums, double bass, warm saxophone` |
| rock | `modern rock, distorted electric guitar, powerful bass, energetic live drums` |
| electronic | `modern electronic pop, synthesizers, electronic drums, deep bass, polished production` |
| folk | `intimate acoustic folk, acoustic guitar, light percussion, natural warm arrangement` |

---

## 4. 推理参数（完整）

模型：**`acestep-v15-turbo`**（8 步蒸馏版，无 CFG）。

| 参数 | 值 | 说明 |
|---|---|---|
| `task_type` | `cover` | 翻唱/曲风编辑 |
| `src_audio` | 源歌 wav | 待编辑音频 |
| `reference_audio` | 源歌 wav（与 src 相同） | 音色参考 |
| `caption` | `"{style}, preserve the original {voice} singer and vocal identity, preserve the original melody, lyric order, pauses, and structure"` | 风格 + 保持指令 |
| `lyrics` | `03_corrected_lyrics/*.txt` | 校正歌词 |
| `vocal_language` | `zh` / `en` | |
| `duration` | 120.0 / 80.0 / 120.0 | 与源歌对齐 |
| **`audio_cover_strength`** | **0.8** | 去噪步数占比，锚**旋律骨架** |
| **`cover_noise_strength`** | **0.6（主结果）/ 0.0（对照）** | 扩散起点，锚**人声波形** |
| `inference_steps` | 8 | |
| `guidance_scale` | 1.0 | turbo 无 CFG |
| `thinking` | False | 关闭 LM 思考 |
| `seed` | `20264000 + shard_index*100 + case_index` | 4 卡 shard 分片 |

### 两个关键旋钮（源码 + 实测确认）

| 参数 | 作用位置 | 管什么 | 结论 |
|---|---|---|---|
| `cover_noise_strength` | 扩散起点（初始 latent） | 锚**人声波形**（歌词、咬字、精确时序） | **主旋钮**，甜点 0.5~0.6；0=纯噪声起步（等于不做 cover），1=最接近源 |
| `audio_cover_strength` | 去噪步数占比 | 锚**旋律骨架**（音符走向、和弦、节奏） | 次要旋钮；**拉满也救不了歌词** |

生成脚本：`run_corrected_full_matrix.py`（ACE-Step-1.5 根目录，4 卡 shard，`--shard-index 0..3`）。

---

## 5. ASR 方法

模型：**Qwen3-ASR-1___7B**（`Qwen3ASRModel`，`bfloat16`，`max_new_tokens=2048`）
路径：`/cfs/cfs-h80qwxi5/kkszhang/FunASR/models/qwen3/Qwen/Qwen3-ASR-1___7B`

| 用途 | 输出目录 | 是否加对齐器 | 输出内容 |
|---|---|---|---|
| 源音频 ASR（歌词基准） | `02_asr_aligned/` | 是（`Qwen3-ForcedAligner-0___6B`） | `text` + `tokens`（逐字时间戳）+ `lines`（按句切分） |
| Cover 音频 ASR | `05_asr_covers/` | 否（只取文本） | `text` + `language` |

- 语言按文件名指定：中文 `Chinese`，英文 `English`。
- 源 ASR 脚本：`transcribe_lyrics.py`；Cover ASR 脚本：`asr_covers.py`。

---

## 6. 覆盖率计算方法

**基准**：源音频的 ASR 文本（`02_asr_aligned/*.json` 的 `text`），代表"原来唱的歌"。

**流程**：Cover 的 ASR 文本 vs 源音频 ASR 文本 → 对齐 → 统计覆盖率。

- 分词：中文按**汉字**（`[\u4e00-\u9fff]`）；英文按**词**（转小写 + 去标点 + 空格切分）。
- 对齐：`difflib.SequenceMatcher`（`autojunk=False`）。
- 指标：
  - **覆盖率** = 匹配 token 数 / 源 ASR token 数；
  - **丢字** = 源有而 Cover 无；
  - **多余** = Cover 有而源无（重复/幻觉/音近错字）。

比对脚本：`compare_cover_vs_source.py`。

```bash
python3 compare_cover_vs_source.py 05_asr_covers 02_asr_aligned
```

---

## 7. 结果

### 7.1 基线：源音频 ASR vs 校正歌词（ASR 本身的误听上限）

| 中文女声 | 英文女声 | 英文男声 |
|---|---:|---:|
| 100% | 96.3% | 91.9% |

> 说明：ASR 对英文"唱"本身有 ~8% 误听，因此覆盖率**不能拿 100% 当及格线**。

### 7.2 主结果：`cover_noise_strength=0.6` / `audio_cover_strength=0.8`

**歌词稳，但曲风接近原唱。**

| 歌 | jazz | rock | electronic | folk |
|---|---:|---:|---:|---:|
| 中文女声 | 98.8% | 99.4% | 97.6% | 95.2% |
| 英文女声 | 82.7% | 88.9% | 92.6% | 75.3% |
| 英文男声 | 75.7% | 75.7% | 59.5% | 73.0% |

### 7.3 对照：`cover_noise_strength=0.0` / `audio_cover_strength=0.8`

**曲风变化大，但丢词严重。**

| 歌 | jazz | rock | electronic | folk |
|---|---:|---:|---:|---:|
| 中文女声 | 86.7% | 75.8% | 67.9% | 63.0% |
| 英文女声 | 54.3% | 66.7% | 61.7% | 44.4% |
| 英文男声 | 43.2% | 29.7% | 45.9% | 45.9% |

### 7.4 结构性结论（论文问题定义来源）

1. **曲风变化 vs 歌词保真，被绑死在同一个物理量上**（"离源多近"）。模型没有独立的曲风/唱法强度通道。
2. 真实短板（正确配置下才暴露）：
   - **句首 onset**：每句第一个词糊掉（`O beautiful` → `for you to fall`）。
   - **生僻词/低频词**：`brotherhood`、`majesties`、`fruited`、中文叠词「慢慢」。
   - 中间常见句基本稳。
3. 人声和伴奏在**同一 latent** 里纠缠，无法"只留人声、只换伴奏"。

---

## 8. 本包目录结构

```text
cover_test_kit/
├── README_测试方法与复现.md          # 本文档
├── sources/                          # 3 首源音频 + 中文歌词
│   ├── chinese_female_120s.wav
│   ├── chinese_female_lyrics.txt
│   ├── english_female_80s.wav
│   └── english_male_120s.wav
├── 03_corrected_lyrics/              # 校正歌词（喂给模型的文本）
│   ├── chinese_female.txt
│   ├── english_female.txt
│   └── english_male.txt
├── 04_corrected_covers/              # 主结果：cns=0.6 的 12 个 Cover
│   ├── chinese_female_{jazz,rock,electronic,folk}.wav
│   ├── english_female_{jazz,rock,electronic,folk}.wav
│   ├── english_male_{jazz,rock,electronic,folk}.wav
│   └── metadata_shard{0..3}.json     # 每条的 source/lyrics/seed
├── 02_asr_aligned/                   # 源音频 ASR（歌词基准，含时间戳）
│   ├── chinese_female_120s.json
│   ├── english_female_80s.json
│   └── english_male_120s.json
├── 05_asr_covers/                    # 12 个 Cover 的 ASR 文本
│   └── {song}_{style}.json
└── scripts/
    ├── compare_cover_vs_source.py    # 覆盖率比对
    ├── compare_lyrics.py             # Cover ASR vs 校正歌词（逐行丢字）
    ├── asr_covers.py                 # Cover ASR
    └── transcribe_lyrics.py          # 源音频 ASR + 对齐
```

---

## 9. 复现步骤

```bash
# 1) 生成 Cover（4 卡 shard，cns=0.6）
cd ACE-Step-1.5
for i in 0 1 2 3; do
  python run_corrected_full_matrix.py --shard-index $i --shard-count 4 &
done
wait

# 2) Cover ASR（Qwen3-ASR）
cd outputs/multi_song_test
python3 asr_covers.py 04_corrected_covers 05_asr_covers cuda:0

# 3) 覆盖率
python3 compare_cover_vs_source.py 05_asr_covers 02_asr_aligned
```

> 注：`asr_covers.py` / `transcribe_lyrics.py` 依赖 `qwen_asr`（Qwen3ASRModel）包，
> 需在 `FunASR/.qwen3_gpu_venv` 环境（`/cfs/cfs-h80qwxi5/kkszhang/FunASR/.qwen3_gpu_venv/bin/python`）下运行。
