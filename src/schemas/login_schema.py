
from pydantic import BaseModel

# 认证请求模型（静态测试账号 user / 1234，演示环境）
class LoginRequest(BaseModel):
    userId: str
    password: str

class RegisterRequest(BaseModel):
    userId: str
    password: str

class RecoverRequest(BaseModel):
    userId: str
    newPassword: str
