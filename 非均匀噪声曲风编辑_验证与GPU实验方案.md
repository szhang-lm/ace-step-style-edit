# 非均匀噪声曲风编辑 —— 验证结论与 GPU 实验方案

日期：2026-10-13
用途：把已验证的部分、产物、以及待 GPU 验证的 DiT 局部实验，整理成可携带到其他机器执行的方案文档。

---

## 一、研究问题与核心假设

**研究问题**：ACE-Step 这类反演/重加噪曲风编辑使用**全局统一噪声强度**（一个标量 `cover_noise_strength`），导致「曲风变化程度」与「歌词/人声保真」被绑死在同一个旋钮上——曲风改得大，人声歌词就被破坏；噪声小，歌词保住但曲风变化不足。

**核心方法（Training-Free）**：在 latent 空间做**非均匀噪声**——按时间轴区分「人声帧」与「伴奏帧」：

- 人声帧：较低噪声、较晚进入生成轨迹，尽量保留歌词与原唱；
- 伴奏帧：较高噪声、较早重新生成，获得明显曲风变化。

**一句话**：把全局标量 `cover_noise_strength` 换成**逐帧噪声 mask**，人声帧走"保真路径"、伴奏帧走"换曲风路径"。

---

## 二、已完成验证（CPU，2026-10-13）

### 验证 1：latent 上能否区分人声/伴奏 —— ✅ 成立（按时间轴定位）

- ACE-Step VAE 时间对齐：`checkpoints/vae/config.json` 中 `downsampling_ratios=[2,4,4,6,10]`（乘积 1920）、`sampling_rate=48000` → **latent 帧率 = 25 Hz**（每帧 40ms）。
- 中文源歌 120s → latent 3000 帧。
- 用中文 ASR 逐句时间戳（`02_asr_aligned/chinese_female_120s.json` 的 `lines`，8 句）映射到 latent 帧，得到人声帧 mask：**2329/3000 帧 = 77.6%**，7 个演唱区间：

  ```
  [0.64,25.0] [25.2,35.56] [38.56,52.2] [52.24,63.0] [65.04,74.44] [75.76,84.68] [87.68,103.4]  (单位:秒)
  ```

- **结论**：latent 上可以按时间轴干净地定位"人声帧（保护）"vs"伴奏帧（换曲风）"。
- ⚠️ **注意**：这是"时间轴定位"，不是"源分离"。VAE 的 64 通道 latent 中人声与伴奏是纠缠的，**无法只留人声、只删伴奏**，只能按帧选择噪声强度。本方法依赖的正是前者。

### 验证 2（renoise 阶段）：人声低噪声是否保护 latent —— ✅ 数学成立 + 已量化

Flow Matching 的 renoise：`xt = t·noise + (1-t)·src_latents`，故 `|xt − src| = t·|noise − src|`。

| 区域 | 噪声强度 t | latent 破坏距离（实测模拟） |
|---|---|---|
| 人声帧 | 0.4（对应 cns=0.6） | 0.321 |
| 伴奏帧 | 1.0（对应 cns=0） | 1.989 |
| 比值 | | **16%**（= 0.4²，与理论一致） |

- **结论**：低噪声起点让人声区域在 latent 上保留 ~84% 源信息，是"去噪后保真"的必要基础。
- **未验证（去噪阶段）**：DiT 从 `xt` 去噪后人声是否真保真——需 GPU 实测（见第四节）。
- **已有全局实证**：之前 cns 扫档显示，全局 cns=0.6（低噪声）歌词覆盖率 95%+，全局 cns=0（高噪声）掉到 56~80%。即"低噪声→去噪后歌词保真"在**全局层面**已成立，缺的是**局部非均匀**的验证。

---

## 三、产物清单

| 产物 | 路径 | 说明 |
|---|---|---|
| 人声帧 mask | `ACE-Step-1.5/outputs/multi_song_test/07_vocal_masks/chinese_female_vocal_mask.npy` | int8 数组，长度 3000，1=人声帧，0=伴奏帧，25Hz |
| mask 元信息 | 同目录 `chinese_female_vocal_mask_meta.json` | 帧数/人声占比/区间列表 |
| 中文 ASR 时间戳 | `outputs/multi_song_test/02_asr_aligned/chinese_female_120s.json` | `lines` 字段 8 句起止 |
| 源歌 | `outputs/multi_song_test/sources/chinese_female_120s.wav` | 48kHz 立体声 16bit，120s |
| mask 生成脚本 | `outputs/multi_song_test/verify1_make_mask.py` | ASR 时间戳 → mask |
| renoise 保真脚本 | `outputs/multi_song_test/verify2_renoise.py` | VAE encode + renoise（需 GPU，CPU 太慢） |
| （作废）VAD 脚本 | `outputs/multi_song_test/verify1_vocal_mask.py` | 早期 VAD 方案，对唱歌不准，勿用 |

---

## 四、待 GPU 验证：DiT 局部非均匀去噪实验

### 4.1 要验证的核心假设

> **H1**：逐帧非均匀噪声（人声帧低 cns、伴奏帧高 cns）能在"伴奏曲风明显变化"的同时"人声歌词保真"。

> **H2（风险）**：DiT 的 attention 帧间耦合，伴奏帧的高噪声变化会"污染"相邻人声帧，导致人声帧低噪声失效。

H1 成立则方向闭环；H2 成立则需要进一步缓解（如软边界、注意力约束）。

### 4.2 实验设计（3 组对比）

| 组 | 人声帧 cns | 伴奏帧 cns | 期望 |
|---|---|---|---|
| A 基线-保真 | 0.6 | 0.6 | 歌词保（~95%），曲风弱 |
| B 基线-换曲风 | 0.0 | 0.0 | 曲风强，歌词丢（~56-80%） |
| C 实验-非均匀 | 0.6 | 0.0 | 歌词保 + 曲风强（关键组） |

中文女声 × 4 曲风（jazz/rock/electronic/folk）跑一遍即可（小规模先用 jazz + rock 两曲风验证）。

### 4.3 实现要点（改 `modeling_acestep_v15_base.py`）

关键代码在 `generate_audio` 方法，当前 renoise 是**全局标量**（约 1969-1993 行）：

```python
# 现状（全局标量）
if cover_noise_strength > 0.0:
    effective_noise_level = 1.0 - cover_noise_strength   # 标量
    nearest_t = min(t_values, key=lambda x: abs(x - effective_noise_level))
    xt = self.renoise(src_latents, nearest_t, noise)      # 全局统一 t
```

**改造为逐帧 mask**（最小改动，冻结模型权重，纯推理外挂）：

```python
# 目标：传入 cover_noise_strength_mask: shape [frames]，每帧 0~1（1=最保真）
# latent 形状 [batch, frames, channels]（与 prepare_noise 中 src_latents_shape 一致）
t_per_frame = 1.0 - cover_noise_strength_mask            # [frames]
t_per_frame = t_per_frame[None, :, None].to(src_latents.device, src_latents.dtype)  # [1, frames, 1]
xt = t_per_frame * noise + (1.0 - t_per_frame) * src_latents   # 逐帧不同噪声强度
```

**mask 来源**：加载 `07_vocal_masks/chinese_female_vocal_mask.npy`，人声帧设 `cns=0.6`、伴奏帧设 `cns=0.0`，并在人声/伴奏边界做**软过渡**（例如边界两侧各 2~4 帧线性插值，避免硬切引入伪影）。

**其余不动**：`audio_cover_strength`（去噪步数占比）、caption、lyrics、seed 均保持与之前 cover 测试一致（`audio_cover_strength=0.8`、`inference_steps=8`）。

### 4.4 评测指标

1. **歌词覆盖率**：cover ASR vs 源歌词，复用 `compare_cover_vs_source.py`（Qwen3-ASR）。
2. **曲风差异度**：cover 与源歌的 embedding 距离（MERT-v2-FullSong 已有权重，或 CLAP/FAD）。
3. **音色保持**（可选）：说话人/歌手 embedding 相似度。

### 4.5 判据（如何判定 H1 成立）

- **H1 成立**：C 组歌词覆盖率 ≥ A 组的 90%（歌词保住），且曲风差异度 ≥ A 组的 1.5 倍（曲风明显变大）。
- **H1 不成立（H2 成立）**：C 组歌词覆盖率跌到接近 B 组（说明 attention 污染，人声帧被伴奏帧带崩）。

---

## 五、环境要求

| 项 | 要求 |
|---|---|
| GPU | NVIDIA，≥24GB 显存（ACE-Step DiT 2B + VAE + LM） |
| torch | 2.10.0+cu128（见 ACE-Step `requirements.txt`） |
| diffusers | 0.34.0（加载 VAE `AutoencoderOobleck`） |
| transformers | 4.57.x（Qwen3-ASR / LM） |
| 模型权重 | `ACE-Step-1.5/checkpoints/`（acestep-v15-turbo / vae / acestep-5Hz-lm-1.7B / Qwen3-Embedding） |
| ASR | Qwen3-ASR-1___7B（`FunASR/models/qwen3/Qwen/Qwen3-ASR-1___7B`） |

> 注：当前无 GPU 环境因 NFS 磁盘 I/O 极慢（torch 加载需几十分钟），无法跑 DiT 实验，故本方案需在 GPU 环境执行。

---

## 六、下一步（若 H1 验证通过）

1. 上 LLM 规划器：让 ACE-Step 现成的 LM（cover 时本被 skip）读歌词+曲风+源特征，输出**逐帧编辑规划**（哪里保、哪里改、改多强），替代手工 mask。
2. 学习式人声感知模块（进阶）：训练轻量模块输出连续软掩码 + 时变噪声调度。
3. 补"曲风差异度"指标，把 24 个 cover（ACE-Step / YuE2）画成"曲风变化 × 歌词保真"二维图。
