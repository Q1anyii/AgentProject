from pydantic import BaseModel
from datetime import datetime
# 认证请求模型（静态测试账号 user / 1234，演示环境）
class LoginRequest(BaseModel):
    userId: str
    password: str

class RegisterRequest(BaseModel):
    userName: str
    userId: str
    password: str
    createTime: datetime
    updateTime: datetime

class RecoverRequest(BaseModel):
    userId: str
    newPassword: str
