
from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(
    title="Fruit Classification API",
    description="API for fruit classification and freshness detection",
    version="1.0.0"
)

app.include_router(router)


@app.get("/")
def home():
    return {
        "message": "Fruit Classification API is running!"
    }
