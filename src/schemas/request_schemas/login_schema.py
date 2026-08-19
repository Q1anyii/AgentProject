from pydantic import BaseModel
from datetime import datetime
from typing import Optional
# 认证请求模型（MySQL 用户表校验）
class LoginRequest(BaseModel):
    userId: str
    password: str

class RegisterRequest(BaseModel):
    userName: str
    userId: str
    password: str
    # 创建/更新时间由后端生成，前端注册时无需传入
    createTime: Optional[datetime] = None
    updateTime: Optional[datetime] = None

class RecoverRequest(BaseModel):
    userId: str
    newPassword: str
