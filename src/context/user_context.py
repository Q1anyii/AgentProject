import contextvars
from typing import Optional
from datetime import datetime

class CtxUser:
    id: Optional[int]
    user_id: Optional[str]
    password: Optional[str]
    username: Optional[ str]
    create_time: Optional[ datetime]
    update_time: Optional[ datetime]


    def __init__(self, uid, user_id, password, username, create_time, update_time):
        self.id = uid
        # 修复：属性名统一为 snake_case，与类注解一致
        self.user_id = user_id
        self.username = username
        self.create_time = create_time
        self.update_time = update_time
        self.password = password


# 声明上下文变量
user_info_ctx = contextvars.ContextVar("user_info", default=None)

