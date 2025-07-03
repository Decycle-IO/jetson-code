# serial_api_server.py

import serial
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import traceback

# --- Configuration ---
# IMPORTANT: Change this to your Arduino's serial port
SERIAL_PORT = '/dev/ttyACM1'  
BAUD_RATE = 115200
API_HOST = "0.0.0.0" # Listen on all network interfaces
API_PORT = 8001

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Serial Command Server",
    description="An API to send commands to a device over a serial port."
)

# Global variable to hold the serial connection
ser = None

# --- Pydantic Model for Request Body ---
class Command(BaseModel):
    command: str

# --- FastAPI Events (Startup and Shutdown) ---
@app.on_event("startup")
def startup_event():
    """
    Initializes the serial connection when the server starts.
    """
    global ser
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
        print(f"Successfully connected to serial port {SERIAL_PORT} at {BAUD_RATE} baud.")
    except serial.SerialException as e:
        print(f"FATAL: Could not open serial port {SERIAL_PORT}: {e}")
        print("Server will start, but requests will fail. Please check the port and permissions.")
        ser = None # Ensure ser is None if connection failed
        traceback.print_exc()

@app.on_event("shutdown")
def shutdown_event():
    """
    Closes the serial connection when the server shuts down.
    """
    if ser and ser.is_open:
        ser.close()
        print(f"Serial port {SERIAL_PORT} closed.")

# --- API Endpoint ---
@app.post("/send_command")
def send_command(payload: Command):
    """
    Receives a command and sends it over the serial port.
    The payload should be a JSON object like: {"command": "1"}
    """
    if ser is None or not ser.is_open:
        raise HTTPException(
            status_code=503, 
            detail=f"Serial port {SERIAL_PORT} is not available. Check server logs."
        )

    try:
        # The original script sent bytes, so we encode the command string to bytes
        cmd_byte = payload.command.encode('utf-8')
        ser.write(cmd_byte)
        print(f"Sent command '{payload.command}' ({cmd_byte}) to serial device.")
        return {"status": "success", "command_sent": payload.command}
    except Exception as e:
        print(f"Failed to send serial command: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to send command to serial port. Reason: {e}"
        )

@app.get("/")
def read_root():
    return {"message": "Serial API Server is running. POST to /send_command to send data."}

# --- Main execution block ---
if __name__ == "__main__":
    print(f"Starting Serial API server on http://{API_HOST}:{API_PORT}")
    uvicorn.run(app, host=API_HOST, port=API_PORT)