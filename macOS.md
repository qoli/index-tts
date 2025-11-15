# IndexTTS macOS (Apple Silicon) 開發環境最佳實踐

*給 Codex 的改善指導文件*

本文件旨在指導 Codex 對 macOS（尤其是 Apple Silicon M3 / M2 / M1）上運行 IndexTTS 的相容性與穩定性進行優化。
重點解決：Bus Error、Segmentation Fault、依賴衝突、PyTorch MPS 崩潰與環境污染等問題。

---

## 一、核心原則（Codex 執行改善時必須遵守）

1. **永遠使用虛擬環境（venv / uv venv），禁止系統 Python。**
2. **所有依賴必須透過 `uv` 管理。不要使用 pip / conda 交叉混用。**
3. **大模型在 macOS 需要完整的 MPS（Metal Performance Shaders）優化環境變數。**
4. **所有 PyTorch 安裝需經 map_location 檢查與 CPU → MPS 再搬移流程。**
5. **能重建環境就永遠重建，不要嘗試 patch 舊環境。**

---

## 二、環境建置流程（Codex 必須依此生成後續任務腳本）

### 1. 使用 uv 建立乾淨環境

```bash
rm -rf .venv
uv venv
uv sync
```

### 2. 安裝 PyTorch（以 uv 保障二進位相容性）

```bash
uv add torch --extra-index-url https://download.pytorch.org/whl/cpu
```

> 原因：官方 wheel 在 macOS Sequoia + 新版 MPS 上更相容，不會觸發 Segmentation Fault 11。

---

## 三、Apple Silicon 必備環境變數（Codex 在所有啟動腳本中必須加入）

```bash
# MPS fallback
export PYTORCH_ENABLE_MPS_FALLBACK=1

# 避免 MPS 記憶體碎片
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# 避免 OpenMP/MKL 衝突
export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=10
export OPENBLAS_NUM_THREADS=10
export VECLIB_MAXIMUM_THREADS=10
export NUMEXPR_NUM_THREADS=10
export KMP_DUPLICATE_LIB_OK=TRUE

# 避免 Bus Error（fork 安全性）
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

# Tokenizers + Transformers 優化
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
```

Codex 在任何自動生成的啟動腳本、systemd、bash script 都必須自動插入上述配置。

---

## 四、模型載入優化（Codex 的 Python 代碼必須遵守）

IndexTTS 與任何大模型在 macOS 需要：

### 1. 統一 CPU → MPS 的 map_location 流程

```python
checkpoint = torch.load(model_path, map_location="cpu")
model.load_state_dict(checkpoint)
model = model.to("mps")
model.eval()
torch.set_grad_enabled(False)
```

### 2. 若發生 MPS 不支援某 op → 自動 fallback

```python
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
```

### 3. 大模型記憶體處理

```python
gc.collect()
if torch.backends.mps.is_available():
    torch.mps.empty_cache()  # 觸發 MPS GC
```

---

## 五、依賴版本衝突的統一解法（Codex 必須在解決方案中使用）

### ❌ 錯誤方式：

* 重新安裝 pip 版本的 protobuf / numpy / httpx
* 混用 conda + pip + uv
* 手動 patch torch 的 dylib

### ✅ 正確方式：

#### 1. 統一所有依賴至 uv lock

```bash
uv lock --upgrade
uv sync
```

#### 2. 強制 protobuf 安裝至兼容版本

```bash
uv add "protobuf==4.25.8" --force
```

#### 3. 若遇到 httpx 衝突，必須確保啟用 venv

```bash
source .venv/bin/activate
```

---

## 六、PyTorch Segmentation Fault 的標準修復流程（Codex 必須遵循）

1. 刪除整個 `.venv`
2. 使用 `uv venv` 重新建立乾淨環境
3. 使用 `uv add torch` 安裝正確 wheel（避免 pip 污染）
4. 添加所有 Apple Silicon 環境變數
5. 在 CPU 載入後再移到 MPS
6. 若仍出現 segfault → Codex 必須重新生成依賴與 pyproject.toml

---

## 七、pyproject.toml 參考模板（Codex 產生新專案必用）

```toml
[project]
name = "index-tts-macos"
requires-python = ">=3.10"

dependencies = [
    "torch>=2.1,<2.5",
    "transformers>=4.30",
    "numpy>=1.24,<2.0",
    "protobuf==4.25.8",
]

[tool.uv.sources]
torch = [
    { index = "pytorch", marker = "sys_platform == 'darwin'" }
]

[[tool.uv.index]]
name = "pytorch"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
```

---

## 八、啟動腳本模板（Codex 必須生成此格式）

```bash
#!/bin/bash
set -e

# Apple Silicon 最佳化配置
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=10
export KMP_DUPLICATE_LIB_OK=TRUE
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

# 啟動 IndexTTS
uv run webui.py
```

---

## 九、Codex 應改善的方向

1. 自動重建環境
2. 自動生成 Apple Silicon 最佳化腳本
3. 自動檢查 Segmentation Fault 來源並強制切換至 uv workflow
4. 為大模型自動加入 CPU → MPS 安全載入流程
5. 自動生成 pyproject.toml 並鎖定版本
6. 一鍵診斷（httpx / protobuf / torch / numpy）版本衝突
7. 自動執行 `torch.backends.mps.is_available()` 確認
8. 自動給出模型落地建議，例如 FP16、Lazy Loading、MPSCaching 等

---

## 十、Codex 生成的任務建議（可直接執行）

* `generate_macos_env_fix.sh`：一鍵修復環境
* `generate_macos_uv_setup.sh`：完整 uv 建置腳本
* `generate_model_loader.py`：安全載入 IndexTTS 模型
* `generate_dependency_diagnose.py`：依賴衝突檢測
* `generate_webui_starter.sh`：啟動與環境變數載入

本文件可作為 Codex 的「提升 IndexTTS macOS 相容性」基準規範。

---
