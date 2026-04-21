# Web Training Console

项目现在提供了一个基于 FastAPI 的训练可视化页面，用来管理 `train.run` 的训练任务。

## 安装依赖

```bash
pip install -r requirements.txt
```

新增依赖：

- `fastapi`
- `uvicorn`
- `jinja2`

## 启动方式

在项目根目录执行：

```bash
uvicorn webapp.main:app --host 127.0.0.1 --port 8000
```

默认访问地址：

```text
http://127.0.0.1:8000
```

## 已支持能力

- 可视化填写训练参数
- 启动后台训练任务
- 轮询查看训练状态、日志、epoch 进度
- 浏览 `train/artifacts/` 下的最近训练运行
- 查看每次运行的训练曲线图与关键指标

## 说明

- Web 控制台底层仍然调用 `python -m train.run`
- 为避免网页日志被 `tqdm` 控制字符污染，训练脚本新增了 `--disable-tqdm`
- 每次从页面发起训练时，都会创建新的 `train/artifacts/run_*` 目录
