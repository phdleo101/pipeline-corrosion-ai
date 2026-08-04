# 部署指南：GitHub + HuggingFace Spaces

> 本指南分两部分：先将代码推送到 GitHub，再部署到 HuggingFace Spaces 获得在线 Demo。

---

## 第一部分：推送到 GitHub

### Step 1: 注册 GitHub 账号

1. 打开 https://github.com
2. 点击右上角 "Sign up"
3. 填写邮箱、密码、用户名，完成注册
4. 验证邮箱（检查收件箱）

### Step 2: 创建 Personal Access Token（PAT）

Git 推送代码时需要用 Token 代替密码验证：

1. 登录 GitHub 后，点击右上角头像 → **Settings**
2. 左侧菜单最底部 → **Developer settings**
3. 点击 **Personal access tokens** → **Tokens (classic)**
4. 点击 **Generate new token** → **Generate new token (classic)**
5. 填写：
   - **Note**: `pipeline-corrosion-ai`（随便写，标记用途）
   - **Expiration**: 选择 90 days
   - **Select scopes**: 勾选 `repo`（完整仓库权限）
6. 点击页面底部 **Generate token**
7. **立即复制 Token**（页面关闭后无法再看到）
   - 格式类似：`ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`
   - 保存到安全的地方

### Step 3: 创建 GitHub 仓库

1. 登录 GitHub，点击右上角 **+** → **New repository**
2. 填写：
   - **Repository name**: `pipeline-corrosion-ai`
   - **Description**: `管道腐蚀预测与标准问答系统 | FDE 跨行业作品集项目一`
   - **Visibility**: 选择 **Public**（公开，面试官需要查看）
   - **Initialize this repository with**: 全部不勾选（我们已有本地仓库）
3. 点击 **Create repository**
4. 复制仓库地址，格式为：
   `https://github.com/你的用户名/pipeline-corrosion-ai.git`

### Step 4: 配置 Git 用户信息

打开终端（Git Bash / PowerShell），执行：

```bash
cd D:\22500\FDE_PROJECT\pipeline-corrosion-ai

# 设置用户名和邮箱（替换为你的信息）
git config user.name "你的名字"
git config user.email "你的邮箱@example.com"
```

### Step 5: 推送代码到 GitHub

```bash
# 添加远程仓库（替换 URL 中的用户名）
git remote add origin https://github.com/你的用户名/pipeline-corrosion-ai.git

# 重命名分支为 main（GitHub 默认）
git branch -M main

# 推送代码
git push -u origin main
```

**此时会弹出验证窗口或提示输入凭据：**
- **Username**: 输入你的 GitHub 用户名
- **Password**: 粘贴刚才生成的 Personal Access Token（不是 GitHub 密码）

推送成功后，访问 `https://github.com/你的用户名/pipeline-corrosion-ai` 即可看到代码。

### Step 6: 优化 GitHub Profile

1. 在 GitHub 创建一个与你用户名同名的仓库（如用户名为 `zhangsan`，则创建 `zhangsan/zhangsan`）
2. 在该仓库创建 `README.md`，内容如下：

```markdown
# 你好，我是 你的名字

## FDE 跨行业 AI 作品集

| # | 项目 | 行业 | Demo | 代码 |
|---|------|------|------|------|
| 1 | 管道腐蚀预测 + 标准问答 | 能源 | [Demo](待填写) | [Code](https://github.com/你的用户名/pipeline-corrosion-ai) |
| 2 | 智能问诊助手 | 医疗 | 待开发 | 待开发 |
| 3 | 电商智能客服 | 零售 | 待开发 | 待开发 |

## 核心能力
- 跨行业快速学习：用 AI 在 3 天内理解陌生行业
- AI 解决方案设计：从模糊需求到可部署架构
- AI 工具编排：Dify / LangChain / Gradio 组合使用
```

3. 将三个项目仓库 **Pin** 到 Profile 首页（点击 "Customize your pins"）

---

## 第二部分：部署到 HuggingFace Spaces

### Step 7: 注册 HuggingFace 账号

1. 打开 https://huggingface.co
2. 点击右上角 "Sign Up"
3. 用 GitHub 账号注册（更方便）或邮箱注册
4. 验证邮箱

### Step 8: 创建 HuggingFace Access Token

1. 登录后点击右上角头像 → **Settings**
2. 左侧菜单 → **Access Tokens**
3. 点击 **New token**
4. 填写：
   - **Name**: `space-deploy`
   - **Role**: 选择 **Write**
5. 点击 **Create**
6. **复制 Token**（格式类似：`hf_xxxxxxxxxxxxxxxxxxxxx`）

### Step 9: 创建 HuggingFace Space

1. 点击右上角头像 → **New Space**
2. 填写：
   - **Space name**: `pipeline-corrosion-ai`
   - **License**: MIT
   - **SDK**: 选择 **Gradio**
   - **Visibility**: **Public**
3. 点击 **Create Space**
4. 页面会显示 Space 的 Git 地址，格式为：
   `https://huggingface.co/spaces/你的用户名/pipeline-corrosion-ai`

### Step 10: 推送代码到 HuggingFace Space

```bash
cd D:\22500\FDE_PROJECT\pipeline-corrosion-ai

# 添加 HuggingFace 远程仓库
git remote add space https://huggingface.co/spaces/你的用户名/pipeline-corrosion-ai

# 推送代码
git push space main
```

**验证提示：**
- **Username**: 输入你的 HuggingFace 用户名
- **Password**: 粘贴 Step 8 中创建的 HuggingFace Token

### Step 11: 等待自动构建

1. 推送完成后，回到 Space 页面
2. 页面会显示 "Building" 状态（正在安装依赖和构建）
3. 等待 2-5 分钟，状态变为 "Running"
4. 构建日志在页面右下角 "Logs" 中查看

### Step 12: 获取在线 Demo 链接

构建成功后，Space 页面顶部会显示：

```
https://你的用户名-pipeline-corrosion-ai.hf.space
```

这就是你的在线 Demo 地址，可以：
- 在简历中添加此链接
- 在 GitHub README 中更新 Demo 链接
- 面试时直接打开展示

---

## 常见问题排查

### Q: git push 时报错 "Authentication failed"

**原因**: Token 过期或权限不足

**解决**:
1. 确认使用的是 Personal Access Token，不是 GitHub 密码
2. 确认 Token 勾选了 `repo` 权限
3. 重新生成 Token 并重试

### Q: HuggingFace Space 构建失败

**原因**: 依赖安装失败或代码错误

**解决**:
1. 查看 Space 页面的 "Logs" → "Build logs"
2. 常见问题：
   - `ModuleNotFoundError`: 检查 requirements.txt 是否包含缺失的包
   - `FileNotFoundError`: 检查文件路径是否正确
   - `Port already in use`: HuggingFace 自动分配端口，无需手动设置

### Q: HuggingFace Space 运行缓慢

**原因**: 首次访问时模型需要训练

**解决**:
- 第一次预测会比较慢（约 5-10 秒），因为模型在内存中训练
- 后续预测会正常速度（< 1 秒）

### Q: 如何更新已部署的代码

```bash
# 修改代码后，提交并推送
cd D:\22500\FDE_PROJECT\pipeline-corrosion-ai
git add -A
git commit -m "update: 更新说明"
git push origin main      # 推送到 GitHub
git push space main       # 推送到 HuggingFace
```

### Q: 如何配置 Dify API 启用智能问答

1. 注册 https://dify.ai 账号
2. 创建知识库应用，上传 NACE/API/ASME 标准文档
3. 在应用「访问 API」页面获取 API URL 和 API Key
4. 编辑 `app_config.yaml`：
   ```yaml
   dify_api_url: "https://api.dify.ai/v1"
   dify_api_key: "app-xxxxxxxxxxxxxxxx"
   ```
5. 或者设置环境变量（HuggingFace Space 的 Settings → Variables）：
   - `DIFY_API_URL` = `https://api.dify.ai/v1`
   - `DIFY_API_KEY` = `app-xxxxxxxxxxxxxxxx`
6. 重新推送代码，问答模块自动升级为智能问答模式

---

## 部署完成检查清单

- [ ] GitHub 仓库创建完成，代码已推送
- [ ] GitHub Profile README 已创建，包含项目链接
- [ ] HuggingFace Space 创建完成
- [ ] 代码已推送到 HuggingFace Space
- [ ] Space 构建成功，状态为 Running
- [ ] 在线 Demo 可正常访问
- [ ] 腐蚀预测功能正常
- [ ] 标准问答功能正常（至少降级模式可用）
- [ ] README 中的 Demo 链接已更新
