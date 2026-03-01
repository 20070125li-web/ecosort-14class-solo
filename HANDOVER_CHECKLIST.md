# EcoSort 交付清单（2026-03-01）

## 1) 代码仓最小交付范围
保留并维护以下目录/文件：

- `backend/`, `src/`, `experiments/`, `configs/`, `scripts/`, `docs/`
- `README.md`, `QUICK_START.md`, `environment.yml`, `requirements.txt`
- `streamlit_demo.py`, `ecosort_mapping.yaml`

不纳入 Git 的内容：

- `data/raw`, `data/proc`, `checkpoints`, `logs`, `wandb`
- 临时状态文件、本地缓存、私有密钥

## 2) 发布前检查

### 代码状态
- [ ] `git status` 无异常大文件（>5MB）待提交
- [ ] 无敏感信息（token、secret、绝对路径）
- [ ] `README.md` 与当前目录结构一致

### 训练/推理可复现
- [ ] 至少一个训练配置可直接运行（如 `configs/efficientnet_b3.yaml`）
- [ ] API 可启动：`python backend/app.py`
- [ ] Demo 可启动：`streamlit run streamlit_demo.py`

### 数据与模型
- [ ] `checkpoints/` 中保存的是完整状态（含 optimizer/scheduler/epoch/best）
- [ ] 关键结果已执行 `dvc add checkpoints/`（必要时包括 `data/`）
- [ ] 执行 `dvc push` 或在 release 提供模型下载地址

## 3) 推荐发布命令
```bash
git add -A
git commit -m "chore: repository cleanup and delivery docs update"
git push origin main
```

若需同步 DVC 产物：
```bash
dvc add checkpoints/
dvc add data/
git add checkpoints.dvc data.dvc .gitignore
git commit -m "chore: track data and checkpoints with dvc"
git push origin main
dvc push
```

## 4) 接手方启动命令
```bash
conda env create -f environment.yml
conda activate ecosort
pip install -r requirements.txt
```

```bash
python backend/app.py
streamlit run streamlit_demo.py
```

## 5) 交付结论
当前仓库应以“可运行代码 + 可复现配置 + 必要文档”为主，训练过程数据和权重通过 DVC/Release 交付，不直接塞入 Git 仓库。
