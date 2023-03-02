from fastapi import FastAPI, Form, File, UploadFile

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello There"}
