
from fastapi import Depends, FastAPI
import uvicorn
from app.routers import category
app = FastAPI()


app.include_router(
    category.router
)


@app.get("/")
async def root():
    return {"message": "Home Page"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)