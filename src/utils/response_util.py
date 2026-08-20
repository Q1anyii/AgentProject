# ============================================================
# 统一响应工具类
# 作用：封装 FastAPI JSONResponse，提供统一的成功/失败响应格式
# 原位置：temp/response_temp.py（已迁移至 utils/ 规范目录）
# ============================================================

from fastapi.responses import JSONResponse


class Response:
    """统一 API 响应封装类，提供 success / failed 两个静态方法。

    成功响应格式：{"ok": true, "message": [...]} 或 {"ok": true}
    失败响应格式：{"ok": false, "message": "..."}，HTTP 状态码 400
    """

    @staticmethod
    def success(*args):
        """成功响应。

        Args:
            *args: 可变参数，作为 message 字段的内容列表；
                   不传参时返回 {"ok": true}，传参时返回 {"ok": true, "message": [args...]}

        Returns:
            JSONResponse: HTTP 200 的 JSON 响应
        """
        content = {"ok": True, "message": args} if args else {"ok": True}
        return JSONResponse(
            content,
            status_code=200,
        )

    @staticmethod
    def failed(message):
        """失败响应。

        Args:
            message: 失败原因描述字符串

        Returns:
            JSONResponse: HTTP 400 的 JSON 响应，格式 {"ok": false, "message": "..."}
        """
        return JSONResponse(
            {"ok": False, "message": message},
            status_code=400,
        )
