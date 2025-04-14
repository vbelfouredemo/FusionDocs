from fastapi import FastAPI, UploadFile, File
import shutil, os

app = FastAPI()
UPLOAD_DIR = "/data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload")
def upload(file: UploadFile = File(...)):
    dest = os.path.join(UPLOAD_DIR, file.filename)
    with open(dest, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"message": f"Uploaded {file.filename}"}