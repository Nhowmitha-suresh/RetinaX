import uvicorn
from server import app

def main():
    print("[*] Launching RetinaX Server...")
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)

if __name__ == "__main__":
    main()
