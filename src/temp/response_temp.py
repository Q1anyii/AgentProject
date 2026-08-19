from fastapi.responses import JSONResponse

class Response:

    @staticmethod
    def success(message):
        return JSONResponse(
                {"ok": True, "message": message},
                status_code=200,
            )

    @staticmethod
    def success():
        return JSONResponse(
            {"ok": True},
            status_code=200,
        )

    @staticmethod
    def failed(message):
        return JSONResponse(
            {"ok": False, "message": message},
            status_code=400,
        )