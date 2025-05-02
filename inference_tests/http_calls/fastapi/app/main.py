# --- 📦 Imports ---
from fastapi import FastAPI
from router import router, INSTANCE_ID  # <-- importamos el router y el UUID

# --- 🚀 Create FastAPI instance ---
app = FastAPI(
    title="Cats vs Dogs Inference API",
    description=f"Inference API Instance ID: {INSTANCE_ID}",  # ¡Incluso puedes ponerlo en la descripción!
    version="1.0.0"
)

# --- 🔗 Include Routers ---
app.include_router(router)

# --- 🎯 Root endpoint (optional) ---
@app.get("/")
def read_root():
    return {"message": "Cats vs Dogs Inference API is running.", "instance_id": INSTANCE_ID}
