# Web App 启动说明

## 1. 后端

在项目根目录执行：

```bash
uvicorn webapp.main:app --host 127.0.0.1 --port 8000
```

默认地址：

```text
http://127.0.0.1:8000
```

接口文档：

```text
http://127.0.0.1:8000/api/docs
```

开发时如果确实需要自动重载，Windows 下建议排除前端构建和依赖目录，避免重载进程卡住导致接口一直无响应：

```bash
uvicorn webapp.main:app --host 127.0.0.1 --port 8000 --reload --reload-exclude frontend/node_modules --reload-exclude frontend/dist
```

## 2. 前端

进入前端目录后执行：

```bash
cd frontend
npm install
npm run dev
```

默认地址：

```text
http://127.0.0.1:5173
```

## 3. 当前已接入能力

- 用户注册 / 登录
- 主页总览
- 单股分析：调用现有预测样本构建与预测模型
- 多股选股：调用现有预测模型进行全市场候选排序
- 数据中心 / 模型中心

## 4. 当前占位能力

- 决策模型
- 大模型最终结论
- 更多扩展功能
