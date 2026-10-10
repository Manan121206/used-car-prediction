# Roadwise - Used Car Valuation

A Streamlit web app for the used-car price prediction project. It loads the supplied trained pipeline, collects the same 20 fields used by that pipeline, and returns a single estimated sale price.

## Run on Windows

Install Python 3.12, open PowerShell in this folder, then run:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Streamlit will print a local address (usually `http://localhost:8501`) and open the site in your browser. Keep the terminal open while using the app.

### Run from VS Code

This project includes a VS Code launch profile named **Run Streamlit app**. Open the project folder, choose that profile in **Run and Debug**, and press **F5**. If VS Code asks for a Python interpreter, select `.venv\Scripts\python.exe`. You can also run this in the integrated terminal:

```powershell
python -m streamlit run app.py
```

Do not use VS Code's plain **Run Python File** action for this app; that launches `app.py` as a normal Python script rather than starting the Streamlit web server. If you rebuild the environment, use Python 3.12 and install `requirements.txt` before running it.

## Project files

- `app.py` - Website, input validation, model prediction, and downloadable estimate.
- `best_used_car_price_model.pkl` - Trained scikit-learn pipeline provided for this project.
- `car_options.json` - Valid make/model/variant combinations derived from the supplied cleaned listings.
- `requirements.txt` - Python packages and the scikit-learn version used to load the model.

The app uses the model's own OneHotEncoder categories for its dropdowns and checks that the loaded model exposes the expected 20 input fields. Make, model, and variant selections are linked so users cannot select a combination absent from the supplied listings.

## Share it online

1. Create a GitHub repository and add `app.py`, `requirements.txt`, `car_options.json`, and `best_used_car_price_model.pkl`.
2. Sign in at [Streamlit Community Cloud](https://share.streamlit.io/) and connect GitHub.
3. Create an app, select the repository and branch, and set `app.py` as the entrypoint.
4. Choose Python 3.12 to match the environment used to verify this project, then deploy.

Do not commit passwords, API keys, or other secrets. The model file is part of the app and must be available to the deployed server.

## Model notes and limits

- Algorithm: `GradientBoostingRegressor` with 300 estimators.
- Input schema: 20 features, including vehicle, location, listing-source, ownership, and date-derived fields.
- Training listings in the provided CSV span 20 February 2019 to 30 May 2021. The app restricts listing dates to this range because newer dates are outside the model's training period.
- The supplied project files do not include MAE, RMSE, or R² results. The app therefore presents a point estimate without claiming a confidence range or accuracy score.
- The UI formats the result with a rupee symbol based on the India-focused locations in the dataset. Confirm that `sale_price` is denominated in INR with the project team before submission.
- For real current-market use, refresh the data, retrain, evaluate, and replace the saved pipeline before widening the listing-date range.

## Verification performed

The app was checked with the saved model under scikit-learn 1.6.1. Verification includes loading the website, checking that the expected form controls render, submitting the default example, confirming that a numeric estimate is produced, and exercising invalid year input.
