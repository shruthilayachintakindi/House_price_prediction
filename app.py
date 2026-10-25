
import joblib
import pandas as pd

from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi import Form

MODEL_PATH = Path(__file__).parent / "artifacts" / "house_price_model.pkl"
model = joblib.load(MODEL_PATH)

app = FastAPI(title="House Price Prediction")


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <head>
        <title>House Price Prediction</title>
        <style>
            body {
                font-family: Arial;
                background: #eef2f7;
                padding: 30px;
            }
            .container {
                max-width: 600px;
                margin: auto;
                background: white;
                padding: 25px;
                border-radius: 12px;
                box-shadow: 0 2px 12px #ccc;
            }
            h1 { text-align: center; color: #234; }
            label { display: block; margin-top: 12px; }
            input, select {
                width: 100%;
                padding: 10px;
                margin-top: 5px;
                box-sizing: border-box;
            }
            button {
                width: 100%;
                margin-top: 20px;
                padding: 12px;
                background: #1769aa;
                color: white;
                border: none;
                border-radius: 6px;
                font-size: 16px;
                cursor: pointer;
            }
        </style>
    </head>
    <body>
      <div class="container">
        <h1>House Price Prediction</h1>
        <form action="/predict" method="post">
          <label>Area (square feet)</label>
          <input type="number" name="area" min="1" step="any" required>

          <label>Bedrooms</label>
          <input type="number" name="bedrooms" min="0" required>

          <label>Bathrooms</label>
          <input type="number" name="bathrooms" min="0" required>

          <label>Stories</label>
          <input type="number" name="stories" min="0" required>

          <label>Main Road</label>
          <select name="mainroad">
            <option>yes</option><option>no</option>
          </select>

          <label>Guest Room</label>
          <select name="guestroom">
            <option>yes</option><option>no</option>
          </select>

          <label>Basement</label>
          <select name="basement">
            <option>yes</option><option>no</option>
          </select>

          <label>Hot Water Heating</label>
          <select name="hotwaterheating">
            <option>yes</option><option>no</option>
          </select>

          <label>Air Conditioning</label>
          <select name="airconditioning">
            <option>yes</option><option>no</option>
          </select>

          <label>Parking Spaces</label>
          <input type="number" name="parking" min="0" required>

          <label>Preferred Area</label>
          <select name="prefarea">
            <option>yes</option><option>no</option>
          </select>

          <label>Furnishing Status</label>
          <select name="furnishingstatus">
            <option value="furnished">Furnished</option>
            <option value="semi-furnished">Semi-furnished</option>
            <option value="unfurnished">Unfurnished</option>
          </select>

          <button type="submit">Predict House Price</button>
        </form>
      </div>
    </body>
    </html>
    """


@app.post("/predict", response_class=HTMLResponse)
def predict_price(
    area: float = Form(...),
    bedrooms: int = Form(...),
    bathrooms: int = Form(...),
    stories: int = Form(...),
    mainroad: str = Form(...),
    guestroom: str = Form(...),
    basement: str = Form(...),
    hotwaterheating: str = Form(...),
    airconditioning: str = Form(...),
    parking: int = Form(...),
    prefarea: str = Form(...),
    furnishingstatus: str = Form(...)
):
    data = pd.DataFrame([{
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "stories": stories,
        "mainroad": mainroad,
        "guestroom": guestroom,
        "basement": basement,
        "hotwaterheating": hotwaterheating,
        "airconditioning": airconditioning,
        "parking": parking,
        "prefarea": prefarea,
        "furnishingstatus": furnishingstatus
    }])

    prediction = float(model.predict(data)[0])

    return f"""
    <html>
    <body style="font-family:Arial;text-align:center;
                 background:#eef2f7;padding:50px">
      <div style="background:white;padding:30px;border-radius:12px;
                  max-width:600px;margin:auto">
        <h1>Predicted House Price</h1>
        <h2 style="color:green">{prediction:,.2f}</h2>
        <p>Prediction is in the same currency units as your training dataset.</p>
        <a href="/">Predict Another House</a>
      </div>
    </body>
    </html>
    """
