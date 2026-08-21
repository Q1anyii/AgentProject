# 编程知识库（Knowledge Base）

> 基于 AgentProject 项目开发过程中的对话历史、代码实践和架构决策总结的编程知识库。
> 涵盖 Python、FastAPI、LangGraph、RAG、数据库、前端、安全、工程化等领域。

## 知识库概述

本知识库包含两部分内容：

1. **知识文档（10篇）**：系统性的技术总结，涵盖架构设计、最佳实践、常见问题
2. **测试 QA（45条）**：按难度分类的问答对，含基础概念、代码调试、架构设计、刁钻 Badcase

所有内容均基于实际项目代码和开发经验，严禁胡编乱造。对不了解的内容未存入。

## 目录结构

```
knowledge-base/
├── README.md                          # 本文件（索引）
├── ingest_knowledge.py                # 向量库入库脚本
│
├── 01-python-best-practices.md        # Python 编程最佳实践
├── 02-fastapi-backend.md              # FastAPI 后端开发实践
├── 03-langgraph-architecture.md       # LangGraph 架构设计实践
├── 04-rag-retrieval-system.md         # RAG 检索系统设计实践
├── 05-database-design.md              # 数据库设计与多存储协作
├── 06-system-architecture.md          # 系统架构设计实践
├── 07-exception-handling.md           # 异常处理机制实践
├── 08-security-authentication.md      # 安全与认证实践
├── 09-frontend-vue3.md                # 前端 Vue 3 开发实践
├── 10-engineering-practices.md        # 工程化实践总结
│
└── test-qa/
    ├── 01-basic-concepts.md           # 基础概念题（10条）
    ├── 02-code-debugging.md           # 代码调试题（10条）
    ├── 03-architecture-design.md      # 架构设计题（10条）
    └── 04-badcase-tricky.md           # 刁钻 Badcase（15条）
```

## 知识文档分类

| 编号 | 文件 | 领域 | 核心内容 |
|------|------|------|----------|
| 01 | Python 最佳实践 | Python | 类型提示、异常处理、异步编程、模块设计、常见陷阱 |
| 02 | FastAPI 后端 | 后端框架 | Lifespan、依赖注入、中间件、全局异常、SSE、文件上传、SPA 托管 |
| 03 | LangGraph 架构 | AI 框架 | 状态图、节点、条件边、Checkpointer、Store、RunnableConfig、流式输出 |
| 04 | RAG 检索系统 | AI 检索 | 向量库、嵌入模型、Query 改写、多查询检索、RRF 融合、重排序、缓存 |
| 05 | 数据库设计 | 数据存储 | MySQL/PostgreSQL/Redis 职责划分、连接管理、表结构、多存储协作 |
| 06 | 系统架构 | 架构设计 | 分层架构、服务层设计、上下文管理、配置管理、常量管理、MCP 集成 |
| 07 | 异常处理 | 可靠性 | 全局异常、业务异常、降级策略、资源清理、错误码设计 |
| 08 | 安全认证 | 安全 | JWT、密码加密、资源归属校验、敏感信息保护、限流、常见漏洞 |
| 09 | 前端 Vue3 | 前端 | Composition API、SSE 接收、AbortController、多主题、文件上传、状态管理 |
| 10 | 工程化实践 | 工程化 | 项目结构、代码规范、日志管理、环境配置、依赖管理、测试、部署、Git |

## 测试 QA 分类

| 编号 | 文件 | 类型 | 数量 | 特点 |
|------|------|------|------|------|
| 01 | 基础概念题 | 概念 | 10条 | 覆盖核心技术概念和原理，适合入门学习 |
| 02 | 代码调试题 | 实战 | 10条 | 给出有 bug 的代码，分析问题并修复，适合代码审查 |
| 03 | 架构设计题 | 设计 | 10条 | 系统设计、技术选型、扩展性、性能优化，适合架构师 |
| 04 | 刁钻 Badcase | 安全/边界 | 15条 | 路径遍历、Prompt 注入、并发冲突、数据泄露等极端场景 |

**总计：45 条 QA**

## 向量库入库

### 方法一：使用入库脚本（推荐）

```bash
cd src
python ../resources/knowledge-base/ingest_knowledge.py
```

脚本会自动遍历 knowledge-base 目录下的所有 .md 文件，调用项目的 `EmbeddingProcessor` 入库。

### 方法二：手动入库

```bash
cd src
python embedding.py
# 输入文件路径：../resources/knowledge-base/01-python-best-practices.md
# 输入 source 和 category：knowledge_base python
```

### 入库元数据

- **source**: `knowledge_base`
- **category**: 根据文件名自动分类（python/fastapi/langgraph/rag/database/architecture/exception/security/frontend/engineering/test_qa）

### 检索使用

入库后，用户提问相关编程问题时，RAG 检索系统会自动从向量库中召回相关知识片段，辅助 AI 回答。

## 内容来源与质量保证

### 内容来源
1. **项目代码**：AgentProject 项目的实际代码实现
2. **对话历史**：开发过程中的技术讨论和问题排查
3. **架构决策**：项目设计中的技术选型和权衡分析
4. **官方文档**：FastAPI、LangGraph、ChromaDB 等官方文档的最佳实践

### 质量保证
- ✅ 所有内容基于实际项目经验，不胡编乱造
- ✅ 代码示例均可运行（基于项目实际代码）
- ✅ 对不了解的内容未存入
- ✅ 个人见解明确标注，与事实区分
- ✅ 持续更新，随项目演进而迭代

## 更新与维护

### 更新原则
1. 项目代码有重大变更时，同步更新相关知识文档
2. 发现新的问题或最佳实践时，新增 QA 条目
3. 定期审查内容，移除过时信息

### 贡献指南
1. 新增知识文档：按 `NN-主题.md` 命名，放入 knowledge-base 根目录
2. 新增 QA：按分类放入 test-qa/ 目录，遵循现有格式
3. 入库：新增文件后运行 `ingest_knowledge.py` 更新向量库

## 免责声明

本知识库仅供学习和参考，不构成任何技术建议。实际项目中请根据具体场景和需求进行评估和决策。
