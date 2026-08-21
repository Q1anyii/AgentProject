# Python 编程最佳实践

> 基于 AgentProject 项目代码总结，涵盖类型提示、异常处理、异步编程、模块设计等核心实践。

## 一、类型提示（Type Hints）

### 1.1 TypedDict 用于结构化字典

当函数参数或返回值是具有固定结构的字典时，使用 `TypedDict` 而非 `dict`，提升可读性和 IDE 补全。

```python
from typing import TypedDict, List, Dict, Any, Optional

class RAGState(TypedDict):
    question: str
    history: List[Dict[str, str]]
    rewritten_queries: List[str]
    merged_docs: List[Any]  # Document 类型
    reranked_docs: List[Any]
    cache_hit: Optional[bool]
```

**适用场景**：LangGraph 状态定义、API 响应结构、配置字典。

### 1.2 Pydantic BaseModel 用于数据校验

对外接口的请求/响应模型使用 Pydantic `BaseModel`，自动校验类型和必填字段。

```python
from pydantic import Field, BaseModel, ConfigDict

class QueryRewriteResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True)  # 允许按别名填充
    main_query: str = Field(..., alias="主查询")
    sub_queries: List[str] = Field(default_factory=list, alias="子查询")
    keywords: List[str] = Field(default_factory=list, alias="关键词")
```

**关键配置**：
- `populate_by_name=True`：允许同时使用字段名和别名初始化
- `Field(..., alias="xxx")`：`...` 表示必填，alias 用于 JSON 键名映射
- `default_factory=list`：可变默认值必须用 factory，避免共享引用

### 1.3 函数签名完整标注

```python
def online_rerank(query: str, documents: list[str], top_n: int = 10) -> list[dict]:
    """调用在线重排，返回按相关性降序的 [{index, relevance_score}, ...]"""
    ...
```

## 二、异常处理

### 2.1 自定义异常类

业务相关的错误应定义自定义异常，而非通用 `Exception`。

```python
class ConfigError(Exception):
    """配置错误异常：缺少必填环境变量或值格式错误时抛出"""
    pass
```

### 2.2 降级策略（Graceful Degradation）

非核心功能失败时，应降级而非中断主流程。

```python
def get_user_system_prompt(user_id: str, base_prompt: str = None) -> str:
    try:
        from service.user_profile_service import user_profile_service
        custom_prompt = user_profile_service.get_system_prompt(user_id)
        if custom_prompt and custom_prompt.strip():
            return f"{base_prompt}\n\n【用户自定义设定】\n{custom_prompt.strip()}"
    except Exception as e:
        # 获取失败时降级使用基础 prompt，不影响对话
        logger.warning(f"获取用户自定义 system prompt 失败 user_id={user_id}: {e}")
    return base_prompt
```

**原则**：
- 核心功能失败 → 抛出异常，快速失败
- 辅助功能失败 → 记录日志，降级处理
- 永远不要用 `except: pass` 吞掉异常

### 2.3 资源清理 with 上下文管理器

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动阶段
    validate_config()
    chat_service.open()
    yield  # 应用运行期间
    # 关闭阶段：确保资源释放
    chat_service.close(timeout=10)
```

## 三、异步编程

### 3.1 asyncio.to_thread 托管阻塞调用

异步函数中调用阻塞 IO（如文件读写、同步 HTTP 请求）时，用 `asyncio.to_thread` 放入线程池，避免阻塞事件循环。

```python
async def aembed(self, file_path, meta) -> int:
    """异步版 embed：阻塞操作放入线程池"""
    return await asyncio.to_thread(self.embed, file_path, meta)
```

### 3.2 ThreadPoolExecutor 并行检索

CPU 密集或 IO 密集的独立任务可用线程池并行执行。

```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=min(len(queries), 4)) as ex:
    raw_results = list(
        ex.map(lambda q: collection.query(query_texts=[q], n_results=TOP_K), queries)
    )
```

**注意**：Python GIL 限制下，线程池适用于 IO 密集任务（如 HTTP 请求、数据库查询），CPU 密集任务应用多进程。

## 四、模块设计

### 4.1 延迟导入避免循环依赖

当两个模块互相依赖时，在函数内部延迟导入，而非模块顶层。

```python
def get_user_system_prompt(user_id: str, base_prompt: str = None) -> str:
    # 延迟导入避免循环依赖：init.py → user_profile_service → init.py
    try:
        from service.user_profile_service import user_profile_service
        ...
    except Exception as e:
        ...
```

### 4.2 模块级单例

服务层对象在模块顶层创建单例，全局复用，避免重复初始化连接。

```python
# service/user_profile_service.py
class UserProfileService:
    def __init__(self):
        self._conn = None
    def open(self): ...
    def close(self): ...

# 模块级单例
user_profile_service = UserProfileService()

# 使用方
from service.user_profile_service import user_profile_service
```

**生命周期管理**：在 FastAPI `lifespan` 中统一 `open()` / `close()`，而非在服务内部自动连接。

### 4.3 常量集中管理

魔法数字、固定字符串提取到 `constant/` 目录，按模块分类。

```python
# constant/retrieval_constants.py
TOP_K = 10
DISTANCE_THRESHOLD = 0.5
RRF_K = 60
REWRITE_PROMPT = """..."""

# constant/embedding_constants.py
COLLECTION_NAME = "mitta_ai_knowledge"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
```

## 五、常见陷阱

### 5.1 可变默认参数

```python
# 错误：所有调用共享同一个 list
def add_item(item, lst=[]):
    lst.append(item)
    return lst

# 正确：用 None 哨兵
def add_item(item, lst=None):
    if lst is None:
        lst = []
    lst.append(item)
    return lst
```

### 5.2 环境变量类型转换

```python
# 错误：环境变量缺失时 int(None) 报错
expire_minutes = int(os.getenv("JWT_EXPIRE_MINUTES"))

# 正确：提供默认值，格式错误时降级
def get_env_int(key: str, default: int) -> int:
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        logger.warning(f"环境变量 {key} 值 '{value}' 不是有效整数，使用默认值 {default}")
        return default
```

### 5.3 字典哈希混入不稳定字段

计算文档 ID 哈希时，只选择稳定的元数据字段（source, file_name, url），不要混入时间戳、随机字段。

```python
def compute_doc_hash_with_meta(docs):
    ids = []
    for doc in docs:
        content = doc.page_content.strip()
        meta_keys = ["source", "file_name", "url"]  # 只选稳定字段
        meta_part = [f"{k}:{doc.metadata.get(k, '')}" for k in meta_keys]
        raw = content + "\n" + "\n".join(meta_part)
        ids.append(hashlib.sha256(raw.encode("utf-8")).hexdigest())
    return ids
```

## 六、个人见解

1. **类型提示不是装饰，是契约**：团队协作中，完整的类型提示能减少 80% 的参数传递错误。建议从项目第一天就强制执行 `mypy` 或 `pyright` 检查。

2. **异常处理的粒度要分层**：底层函数抛出具体异常，中间层可以捕获并添加上下文，顶层（API 层）统一转换为用户友好的错误响应。不要在每一层都 try-except。

3. **延迟导入是技术债信号**：如果频繁需要延迟导入，说明模块划分有问题。理想的依赖方向是：上层依赖下层，同层之间不互相依赖。

4. **单例模式要谨慎**：模块级单例方便但不利于测试。如果需要 mock 单例，考虑用依赖注入容器或在测试中 `monkeypatch`。
