from fastapi.responses import JSONResponse

class Response:

    @staticmethod
    def success(*args):
        content = {"ok": True, "message": args} if args else {"ok": True}
        return JSONResponse(
                content,
                status_code=200,
            )


    @staticmethod
    def failed(message):
        return JSONResponse(
            {"ok": False, "message": message},
            status_code=400,
        )