from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from io import StringIO
from sklearn.linear_model import LinearRegression

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

historical_data = {"numbers": [], "dates": []}

@app.post("/import-csv")
async def import_csv(file: UploadFile = File(...)):
    content = await file.read()
    df = pd.read_csv(StringIO(content.decode('utf-8')))
    
    numbers = []
    dates = []
    for _, row in df.iterrows():
        dates.append(row['date'])
        numbers.extend([int(row['first']), int(row['second']), int(row['third'])])
    
    historical_data["numbers"] = numbers
    historical_data["dates"] = dates
    
    return {"status": "success", "numbers_count": len(numbers)}

@app.get("/analyze-stats")
async def analyze_stats():
    numbers = historical_data["numbers"]
    if not numbers:
        return {"error": "No data available"}
    
    digit_dist = np.zeros((4, 10))
    besar, kecil, ganjil, genap = 0, 0, 0, 0
    
    for num in numbers:
        digits = [int(d) for d in str(num).zfill(4)]
        for pos, digit in enumerate(digits):
            digit_dist[pos][digit] += 1
        last_digit = digits[3]
        besar += 1 if last_digit >= 5 else 0
        kecil += 1 if last_digit < 5 else 0
        ganjil += 1 if last_digit % 2 != 0 else 0
        genap += 1 if last_digit % 2 == 0 else 0
    
    total_entries = len(numbers)
    digit_dist = (digit_dist / total_entries * 100).tolist()
    
    stats = {
        "total": total_entries,
        "besar": (besar / total_entries * 100),
        "kecil": (kecil / total_entries * 100),
        "ganjil": (ganjil / total_entries * 100),
        "genap": (genap / total_entries * 100)
    }
    
    return {
        "stats": stats,
        "digit_distribution": {
            "labels": list(range(10)),
            "datasets": [
                {"label": "AS", "data": digit_dist[0], "backgroundColor": "#3b82f6"},
                {"label": "KOP", "data": digit_dist[1], "backgroundColor": "#ef4444"},
                {"label": "KEPALA", "data": digit_dist[2], "backgroundColor": "#10b981"},
                {"label": "EKOR", "data": digit_dist[3], "backgroundColor": "#f59e0b"}
            ]
        }
    }

@app.get("/predict")
async def predict():
    numbers = historical_data["numbers"]
    if not numbers:
        return {"error": "No data available"}
    
    last_numbers = numbers[-30:] if len(numbers) >= 30 else numbers
    normalized = [[int(d) / 9 for d in str(num).zfill(4)] for num in last_numbers]
    input_data = np.array([np.array(normalized).flatten()])
    
    model = LinearRegression()
    target = np.array(normalized)[:, -1]  # Ambil digit terakhir sebagai target
    model.fit(input_data[:-1], target[:-1])  # Latih model
    
    predictions = []
    for _ in range(3):
        pred = model.predict(input_data[-1].reshape(1, -1))[0]
        pred_digits = [round(pred * 9) for _ in range(4)]  # Sederhana, hanya untuk contoh
        pred_number = ''.join(map(str, pred_digits))
        last_two = int(''.join(map(str, pred_digits[2:]))) % 12
        shio = ["Tikus", "Kerbau", "Macan", "Kelinci", "Naga", "Ular", "Kuda", "Kambing", "Monyet", "Ayam", "Anjing", "Babi"][last_two]
        predictions.append({
            "number": pred_number,
            "shio": shio,
            "ekor": f"{'Besar' if pred_digits[3] >= 5 else 'Kecil'}, {'Genap' if pred_digits[3] % 2 == 0 else 'Ganjil'}"
        })
    
    return {"predictions": predictions}