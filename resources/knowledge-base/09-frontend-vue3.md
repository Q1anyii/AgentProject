# 前端 Vue 3 开发实践

> 基于 AgentProject 项目总结，涵盖 Vue 3 Composition API、SSE 流式接收、AbortController 中断、多主题系统、单文件 SPA 架构等前端实践。

## 一、前端架构

```
┌─────────────────────────────────────────────────────┐
│  index.html (单文件 SPA, ~100KB)                     │
│                                                       │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────┐ │
│  │  CSS 样式    │  │  HTML 模板   │  │  JS 逻辑  │ │
│  │  - 主题变量  │  │  - Vue 模板  │  │  - setup()│ │
│  │  - 组件样式  │  │  - 弹窗 DOM  │  │  - API调用│ │
│  │  - 响应式    │  │  - 菜单结构  │  │  - 状态管理│ │
│  └─────────────┘  └──────────────┘  └───────────┘ │
│                                                       │
│  技术栈：Vue 3 (CDN) + 原生 CSS + Fetch API          │
│  无构建工具，直接浏览器运行                             │
└─────────────────────────────────────────────────────┘
```

**为什么用单文件 SPA**：
- 项目规模适中，单文件便于部署和维护
- 无需构建工具，修改后直接刷新生效
- Vue 3 CDN 引入，Composition API 足够灵活
- 适合快速迭代和原型验证

## 二、Vue 3 Composition API

### 2.1 setup() 函数

```javascript
const { createApp, ref, computed, onMounted, onUnmounted } = Vue;

const app = createApp({
    setup() {
        // ===== 状态定义 =====
        const user = ref(cache.get(STORAGE_KEY.USER, null));
        const messages = ref([]);
        const inputText = ref('');
        const isLoading = ref(false);
        const streaming = ref(false);
        const sidebarOpen = ref(false);

        // ===== 计算属性 =====
        const userAvatar = computed(() => {
            return user.value?.name?.charAt(0).toUpperCase() || 'U';
        });

        const canSend = computed(() => {
            return inputText.value.trim() && !isLoading.value;
        });

        // ===== 方法 =====
        async function sendMessage() { ... }
        function logout() { ... }

        // ===== 生命周期 =====
        onMounted(() => { ... });
        onUnmounted(() => { ... });

        // ===== 返回模板可用的变量和方法 =====
        return {
            user, messages, inputText, isLoading, streaming,
            userAvatar, canSend, sendMessage, logout,
        };
    }
});
```

### 2.2 响应式状态管理

```javascript
// ref：基本类型
const count = ref(0);
count.value++;  // 访问/修改用 .value

// computed：派生状态
const doubleCount = computed(() => count.value * 2);

// ref 对象：复杂类型
const user = ref({ name: '张三', age: 20 });
user.value.name = '李四';  // 修改对象属性
```

### 2.3 模板语法

```html
<!-- 文本插值 -->
<div>{{ user.name }}</div>

<!-- 属性绑定 -->
<img :src="user.avatar" :alt="user.name">

<!-- 条件渲染 -->
<div v-if="isLoading">加载中...</div>
<div v-else>内容</div>

<!-- 列表渲染 -->
<div v-for="msg in messages" :key="msg.id">{{ msg.content }}</div>

<!-- 事件绑定 -->
<button @click="sendMessage" :disabled="!canSend">发送</button>

<!-- 双向绑定 -->
<textarea v-model="inputText" @keydown="handleKeydown"></textarea>

<!-- 类名绑定 -->
<div :class="{ active: sidebarOpen, 'dark-mode': isDark }"></div>
```

## 三、SSE 流式接收

### 3.1 Fetch + ReadableStream

```javascript
async function apiChat(query, threadId, onStream, signal) {
    const response = await fetch(`${API_BASE}/api/chat/`, {
        method: 'POST',
        headers: authHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({ query, thread_id: threadId }),
        signal: signal,  // AbortController 信号
    });

    // 错误处理
    if (response.status === 401) {
        cache.remove(STORAGE_KEY.USER);
        window.location.href = '/api/login';
        throw new Error('登录已过期，请重新登录');
    }
    if (!response.ok) throw new Error(`请求失败: ${response.status}`);

    // SSE 流式读取
    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';
    let answer = '';

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split('\n\n');
        buffer = events.pop();  // 保留不完整的事件
        for (const event of events) {
            const line = event.trim();
            if (!line.startsWith('data:')) continue;
            const payload = line.slice(5).trim();
            if (!payload || payload === '[DONE]') continue;
            try {
                const chunk = JSON.parse(payload);
                const text = extractContentText(chunk.content);
                if (text) {
                    answer += text;
                    if (onStream) onStream(answer);
                }
            } catch {
                continue;  // 跳过损坏的 SSE 数据
            }
        }
    }
    return answer;
}
```

### 3.2 流式渲染优化（节流）

```javascript
let latestText = '';
let renderTimer = null;
let saveTimer = null;

const answer = await apiChat(content, threadId, (text) => {
    streaming.value = true;
    latestText = text;
    // 渲染节流：100ms 渲染一次，避免每个 token 都做 v-html 重渲染
    if (!renderTimer) {
        renderTimer = setTimeout(() => {
            renderTimer = null;
            aiMsg.content = latestText;
            scrollToBottom();
        }, 100);
    }
    // 保存节流：500ms 保存一次，避免频繁 localStorage 序列化
    if (!saveTimer) {
        saveTimer = setTimeout(() => {
            saveTimer = null;
            saveMessages();
        }, 500);
    }
});
```

**为什么需要节流**：
- LLM 流式输出可能每秒几十个 token
- 每个 token 都触发 Vue 重渲染 + localStorage 序列化，会阻塞主线程
- 节流后视觉上仍是平滑逐字输出，但性能大幅提升

### 3.3 SSE 数据解析

```javascript
function extractContentText(content) {
    """从 LangGraph 流式输出中提取文本内容。"""
    if (typeof content === 'string') return content;
    if (Array.isArray(content)) {
        return content.map(item => {
            if (typeof item === 'string') return item;
            if (item.type === 'text') return item.text;
            return '';
        }).join('');
    }
    if (content && typeof content === 'object') {
        return content.text || content.content || '';
    }
    return '';
}
```

## 四、AbortController 中断请求

### 4.1 发起请求时创建 AbortController

```javascript
// 状态
const abortController = ref(null);
const isLoading = ref(false);

async function sendMessage() {
    // ...
    isLoading.value = true;

    // 创建 AbortController
    abortController.value = new AbortController();

    try {
        const answer = await apiChat(
            content, threadId,
            (text) => { /* 流式回调 */ },
            abortController.value.signal  // 传入信号
        );
        // ...
    } catch (err) {
        // 用户主动停止
        if (err.name === 'AbortError') {
            // 保留已生成的部分内容，不删除 AI 消息
            aiMsg.content = aiMsg.content || '（已停止）';
            return;
        }
        // 其他错误处理...
    } finally {
        isLoading.value = false;
        abortController.value = null;
    }
}
```

### 4.2 停止按钮

```html
<!-- 发送按钮 / 暂停按钮切换 -->
<button v-if="!isLoading" class="send-btn" @click="sendMessage" :disabled="!canSend">
    <svg>发送图标</svg>
</button>
<button v-else class="stop-btn" @click="stopResponse" title="停止回复">
    <svg>停止图标</svg>
</button>
```

```javascript
function stopResponse() {
    if (abortController.value) {
        abortController.value.abort();
        abortController.value = null;
    }
    isLoading.value = false;
    streaming.value = false;
    showToast('已停止回复', 'info');
}
```

### 4.3 AbortError 处理

```javascript
catch (err) {
    if (err.name === 'AbortError') {
        // 用户主动停止：保留已生成内容，清理定时器
        if (renderTimer) { clearTimeout(renderTimer); renderTimer = null; }
        if (saveTimer) { clearTimeout(saveTimer); saveTimer = null; }
        aiMsg.content = aiMsg.content || '（已停止）';
        streaming.value = false;
        saveMessages();
        return;
    }
    // 其他错误...
}
```

## 五、多主题系统

### 5.1 CSS 变量主题

```css
/* 默认主题（:root） */
:root {
    --bg-primary: #F7F5F0;
    --bg-secondary: #FFFFFF;
    --text-primary: #1A1A19;
    --text-secondary: #6B6B6B;
    --accent: #3D6B5B;
    --accent-hover: #2D5A4A;
    --border: #E5E2DB;
    --shadow: 0 2px 8px rgba(0,0,0,0.08);
}

/* 深色主题 */
[data-theme="dark"] {
    --bg-primary: #1A1A19;
    --bg-secondary: #2A2A29;
    --text-primary: #F0F0EF;
    --text-secondary: #A0A0A0;
    --accent: #6BA896;
    --accent-hover: #7BB8A6;
    --border: #3A3A39;
    --shadow: 0 2px 8px rgba(0,0,0,0.3);
}

/* 海洋主题 */
[data-theme="ocean"] {
    --bg-primary: #F0F4F8;
    --bg-secondary: #FFFFFF;
    --text-primary: #1A2A3A;
    --accent: #2E6B9E;
    --accent-hover: #1E5B8E;
    --border: #D5DDE5;
}

/* 其他主题... */

/* 主题切换过渡动画 */
* {
    transition: background-color 0.3s ease, color 0.3s ease, border-color 0.3s ease;
}
```

### 5.2 主题切换

```javascript
const themes = [
    { value: 'default', name: '默认', preview: 'linear-gradient(135deg, #3D6B5B, #F7F5F0)' },
    { value: 'dark', name: '深色', preview: 'linear-gradient(135deg, #6BA896, #1A1A19)' },
    { value: 'ocean', name: '海洋', preview: 'linear-gradient(135deg, #2E6B9E, #F0F4F8)' },
    { value: 'sunset', name: '日落', preview: 'linear-gradient(135deg, #C4622E, #FBF5F0)' },
    { value: 'forest', name: '森林', preview: 'linear-gradient(135deg, #2E8B57, #F0F5F0)' },
    { value: 'lavender', name: '薰衣草', preview: 'linear-gradient(135deg, #7B4FA8, #F5F0F8)' },
];

function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    cache.set('theme', theme);  // localStorage 持久化
}

// 初始化时恢复主题
onMounted(() => {
    const savedTheme = cache.get('theme') || 'default';
    if (savedTheme !== 'default') {
        applyTheme(savedTheme);
    }
});
```

### 5.3 主题选择器 UI

```html
<div class="theme-grid">
    <div v-for="t in themes" :key="t.value"
         class="theme-option"
         :class="{ active: settingsForm.theme === t.value }"
         @click="settingsForm.theme = t.value; applyTheme(t.value)">
        <div class="theme-preview" :style="{ background: t.preview }"></div>
        <div class="theme-name">{{ t.name }}</div>
    </div>
</div>
```

## 六、文件上传

### 6.1 多格式文件上传

```html
<input type="file" ref="fileInput" @change="handleFileUpload"
       multiple
       accept=".txt,.md,.csv,.json,.xml,.html,.pdf,.doc,.docx,.ppt,.pptx,.xls,.xlsx,.png,.jpg,.jpeg,.gif,.webp,.py,.js,.ts,.java,.zip,.rar,.7z">
```

```javascript
async function handleFileUpload(event) {
    const files = event.target.files;
    if (!files || files.length === 0) return;

    for (const file of files) {
        if (file.size > 10 * 1024 * 1024) {
            showToast(`文件 ${file.name} 超过 10MB，已跳过`, 'error');
            continue;
        }
        try {
            const formData = new FormData();
            formData.append('file', file);
            if (currentThreadId.value) {
                formData.append('thread_id', currentThreadId.value);
            }
            const res = await fetch(`${API_BASE}/api/chat/upload`, {
                method: 'POST',
                headers: authHeaders({}),  // 不要设 Content-Type，让浏览器自动设置 boundary
                body: formData,
            });
            if (res.ok) {
                const data = await res.json();
                uploadedFiles.value.push({
                    name: file.name,
                    size: file.size,
                    file_id: data.data?.file_id,
                });
                showToast(`文件 ${file.name} 上传成功`, 'success');
            }
        } catch (err) {
            showToast(`文件 ${file.name} 上传失败：${err.message}`, 'error');
        }
    }
    // 清空 input，允许重复上传同一文件
    if (fileInput.value) fileInput.value.value = '';
}
```

**注意事项**：
- FormData 上传时不要手动设置 `Content-Type`，让浏览器自动设置 boundary
- 大文件要在前端先校验大小，避免上传到后端才拒绝
- 上传完成后清空 input value，否则同一文件不能重复上传

## 七、本地存储封装

### 7.1 Cache 工具

```javascript
const cache = {
    get(key, defaultValue = null) {
        try {
            const value = localStorage.getItem(key);
            return value ? JSON.parse(value) : defaultValue;
        } catch {
            return defaultValue;
        }
    },
    set(key, value) {
        try {
            localStorage.setItem(key, JSON.stringify(value));
        } catch (e) {
            console.error('localStorage 写入失败:', e);
        }
    },
    remove(key) {
        localStorage.removeItem(key);
    },
};

// 会话级存储（sessionStorage）
const sessionCache = {
    get(key, defaultValue = null) { ... },
    set(key, value) { ... },
};
```

### 7.2 存储 Key 常量

```javascript
const STORAGE_KEY = {
    USER: 'mitta_user',
    SESSIONS: 'mitta_sessions',
    CURRENT_THREAD: 'mitta_current_thread',
    THEME: 'mitta_theme',
};
```

## 八、常见前端陷阱

### 8.1 Vue 响应式丢失

```javascript
// 错误：直接替换 ref 的值，响应式丢失
messages.value = newMessages;  // 如果 newMessages 是普通对象，可能丢失响应式

// 正确：用 splice 或重新赋值响应式对象
messages.value.splice(0, messages.value.length, ...newMessages);
```

### 8.2 v-for 缺少 key

```html
<!-- 错误：缺少 key，Vue 无法高效更新 -->
<div v-for="msg in messages">{{ msg.content }}</div>

<!-- 正确：用唯一 id 作为 key -->
<div v-for="msg in messages" :key="msg.id">{{ msg.content }}</div>
```

### 8.3 SSE 缓冲区处理

```javascript
// 错误：假设每次 read 都是完整事件
const text = decoder.decode(value);
const event = JSON.parse(text);  // 可能解析失败，因为事件可能不完整

// 正确：用缓冲区累积，按 \n\n 分割
buffer += decoder.decode(value, { stream: true });
const events = buffer.split('\n\n');
buffer = events.pop();  // 保留不完整的事件
```

### 8.4 异步竞态条件

```javascript
// 错误：快速发送多条消息，后发的可能先返回，导致顺序错乱
async function sendMessage() {
    const answer = await apiChat(...);
    aiMsg.content = answer;  // 可能覆盖了后一条消息的内容
}

// 正确：用请求 ID 或禁用发送按钮避免竞态
async function sendMessage() {
    if (isLoading.value) return;  // 防止重复发送
    isLoading.value = true;
    try { ... } finally { isLoading.value = false; }
}
```

### 8.5 内存泄漏

```javascript
// 错误：定时器/事件监听器未清理
onMounted(() => {
    setInterval(() => { ... }, 1000);  // 组件卸载后仍在运行
    document.addEventListener('click', handler);
});

// 正确：在 onUnmounted 中清理
let timer = null;
onMounted(() => {
    timer = setInterval(() => { ... }, 1000);
    document.addEventListener('click', handler);
});
onUnmounted(() => {
    if (timer) clearInterval(timer);
    document.removeEventListener('click', handler);
});
```

## 九、个人见解

1. **单文件 SPA 适合中小项目，大项目需要工程化**：本项目用单文件 Vue 3 SPA，开发效率高、部署简单。但如果项目继续增长（组件超过 20 个、代码超过 2000 行），应该迁移到 Vite + 多组件架构，否则单文件会难以维护。

2. **SSE 流式接收的细节决定体验**：缓冲区处理、节流渲染、错误恢复、AbortController 中断，每一个细节都影响用户体验。很多项目的 SSE 实现只能跑通，但在网络波动、快速发送、停止重试等场景下会出问题。

3. **多主题系统用 CSS 变量是最佳方案**：相比用 class 切换整套样式，CSS 变量更灵活、性能更好、主题数量不受限。关键是设计好变量层级（基础变量 → 组件变量），避免硬编码颜色。

4. **AbortController 是现代前端必备技能**：不仅用于停止 AI 回复，还可以用于组件卸载时取消未完成的请求、路由切换时取消旧页面请求。很多前端 bug（组件卸载后 setState、请求竞态）都可以用 AbortController 解决。

5. **前端状态管理要分层**：本项目用 ref + computed + localStorage 手动管理状态，适合中小项目。但如果状态复杂（多组件共享、跨页面持久化、服务端同步），应该考虑 Pinia 等状态管理库。不要过早引入，但也不要回避必要的抽象。
