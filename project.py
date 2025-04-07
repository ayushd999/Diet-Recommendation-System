from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import ctypes
import os
import tempfile

app = Flask(__name__,
            static_url_path='',
            static_folder=r"C:\Users\Ayush Das\Desktop\diet_recommendation\static",
            template_folder=r"C:\Users\Ayush Das\Desktop\diet_recommendation\templates")

# === Load and prepare data ===
csv_path = r"C:\Users\Ayush Das\Desktop\diet_recommendation\diet_recommendation_dataset.csv"
df = pd.read_csv(csv_path)

encoder = LabelEncoder()
df['Diet Category'] = encoder.fit_transform(df['Diet Category'])

x = df.iloc[:, 1:5]
y = df.iloc[:, 5:]

lg = LogisticRegression(max_iter=1000)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, train_size=0.8, test_size=0.2)
lg.fit(xtrain, ytrain)

# === Load C DLL ===
dll_path = r"C:\Users\Ayush Das\Desktop\diet_recommendation\diett.dll"
lib = ctypes.CDLL(dll_path)
lib.loadCSV.argtypes = [ctypes.c_char_p]
lib.loadCSV.restype = None
lib.sort_carbs.restype = lib.sort_prots.restype = lib.sort_fats.restype = None
lib.carb_rich.restype = lib.protein_rich.restype = lib.fat_rich.restype = None
lib.loadCSV(csv_path.encode())

# === Utility: Capture C output from stdout ===
def capture_output(func):
    """
    Redirects stdout to capture output from a function call.
    Works even for output printed by C functions.
    """
    # Save the current stdout file descriptor
    original_fd = os.dup(1)
    with tempfile.TemporaryFile(mode='w+b') as tmp:
        os.dup2(tmp.fileno(), 1)  # Redirect stdout to temporary file
        try:
            func()
        finally:
            os.dup2(original_fd, 1)  # Restore original stdout
            os.close(original_fd)
        tmp.seek(0)
        output = tmp.read().decode()
    return output

# === Routes ===
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        carbs = float(data["carbs"])
        prots = float(data["proteins"])
        fats = float(data["fats"])
        cals = float(data["calories"])
        res = lg.predict([[carbs, prots, fats, cals]])[0]

        # Define a helper function to call the DLL functions
        def call_c_functions():
            if res == 2:
                lib.sort_prots()
                lib.protein_rich()
            elif res == 1:
                lib.sort_fats()
                lib.fat_rich()
            elif res == 0:
                lib.sort_carbs()
                lib.carb_rich()

        # Capture the output from the C functions
        details = capture_output(call_c_functions)

        if res == 2:
            message = "You need more Protein-rich foods."
        elif res == 1:
            message = "You need more Fat-rich foods."
        elif res == 0:
            message = "You need more Carb-rich foods."
        else:
            message = "No clear recommendation found."

        return jsonify({"message": message, "details": details})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
