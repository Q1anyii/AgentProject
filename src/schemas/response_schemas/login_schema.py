from pydantic import BaseModel
from datetime import datetime
# 认证请求模型（静态测试账号 user / 1234，演示环境）
class LoginResponse(BaseModel):
    id: int
    userId: str
    password: str
    username: str
    createTime: datetime
    updateTime: datetime

class RegisterResponse(BaseModel):
    userName: str
    userId: str
    password: str
    createTime: datetime
    updateTime: datetime

class RecoverResponse(BaseModel):
    userId: str
    newPassword: str
