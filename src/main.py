from urllib.request import localhost

from fastapi import FastAPI

app = FastAPI(title="智能AI客服")

# get接口
@app.get("/")
def root():
    return {"msg":"hello fastapi"}

@app.get("/api/hello/{name}")
def say_hello(name:str):
    return {"name":name}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=localhost, port=8000, reload=True)