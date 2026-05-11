from flask import Flask, render_template, request, redirect, session, send_file
import pandas as pd
import pickle
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__)
app.secret_key = "secret123"

MODEL_FILE = "model.pkl"

# ---------------- GLOBAL STORAGE ----------------
train_accuracies = []
prediction_results = []
history_data = []

# 🔥 IMPORTANT: DEFINE SYMPTOMS (MATCH YOUR FORM ORDER)
symptoms = ["Fever", "Cough", "Headache", "Fatigue"]

# 🔥 DISEASE MAPPING (MATCH YOUR DATASET)
disease_map = {
    0: "Flu",
    1: "Cold",
    2: "Dengue",
    3: "Migraine"
}


# ---------------- LOGIN ----------------
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        session['user'] = email
        return redirect('/dashboard')
    return render_template("login.html")


# ---------------- REGISTER ----------------
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        return redirect('/')
    return render_template("register.html")


# ---------------- DASHBOARD ----------------
@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect('/')
    return render_template("dashboard.html")


# ---------------- TRAIN MODEL ----------------
@app.route('/train', methods=['GET', 'POST'])
def train():
    if 'user' not in session:
        return redirect('/')

    message = ""
    trained = False

    if request.method == 'POST':

        file = request.files.get('file')

        if file and file.filename.endswith('.csv'):

            try:
                df = pd.read_csv(file)

                # CLEAN DATA
                df = df.applymap(lambda x: str(x).strip())

                # CONVERT TEXT → NUMERIC
                for col in df.columns:
                    if df[col].dtype == 'object':
                        df[col] = df[col].astype('category').cat.codes

                # SPLIT
                X = df.iloc[:, :-1]
                y = df.iloc[:, -1]

                # CHECK CLASSES
                if len(y.unique()) < 2:
                    message = "❌ Dataset must contain at least 2 classes"
                    return render_template("train.html", message=message, trained=False)

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

                model = LogisticRegression(max_iter=1000)
                model.fit(X_train, y_train)

                # ACCURACY
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)

                # SAVE MODEL
                with open(MODEL_FILE, "wb") as f:
                    pickle.dump(model, f)

                # STORE FOR GRAPH
                train_accuracies.append(round(acc * 100, 2))

                message = f"✅ Model trained successfully! Accuracy: {round(acc*100,2)}%"
                trained = True

            except Exception as e:
                message = f"❌ Error: {str(e)}"

        else:
            message = "❌ Upload a valid CSV file"

    return render_template("train.html", message=message, trained=trained)


# ---------------- PREDICT ----------------
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'user' not in session:
        return redirect('/')

    result = ""

    if request.method == 'POST':

        try:
            model = pickle.load(open(MODEL_FILE, 'rb'))
        except:
            return render_template("predict.html", result="❌ Train model first")

        n_features = model.n_features_in_

        values = []
        input_text = []

        # 🔥 CONVERT INPUTS TO SYMPTOMS
        for i, key in enumerate(request.form):
            val = request.form.get(key)

            try:
                num = float(val)
            except:
                num = 0

            values.append(num)

            if num == 1:
                input_text.append(symptoms[i])

        # FIX FEATURE SIZE
        while len(values) < n_features:
            values.append(0)

        values = values[:n_features]

        prediction = model.predict([values])[0]

        # 🔥 CONVERT RESULT TO NAME
        result = disease_map.get(int(prediction), str(prediction))

        # STORE FOR GRAPH
        prediction_results.append(result)

        # 🔥 STORE HISTORY (FINAL FIX)
        history_data.append({
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "input": ", ".join(input_text) if input_text else "No symptoms",
            "result": result
        })

    return render_template("predict.html", result=result)


# ---------------- HISTORY ----------------
@app.route('/history')
def history():
    if 'user' not in session:
        return redirect('/')

    return render_template("history.html", history=history_data)

#-------------------- DELETE HISTORY ----------------
# DELETE SINGLE RECORD
@app.route('/delete_history/<int:index>')
def delete_history(index):
    if 'user' not in session:
        return redirect('/')

    if 0 <= index < len(history_data):
        history_data.pop(index)

    return redirect('/history')


# CLEAR ALL HISTORY
@app.route('/clear_history')
def clear_history():
    if 'user' not in session:
        return redirect('/')

    history_data.clear()
    return redirect('/history')


# ---------------- GRAPH ----------------
@app.route('/graph')
def graph():
    if 'user' not in session:
        return redirect('/')

    data = train_accuracies.copy()

    if len(data) == 1:
        data.append(data[0])

    return render_template(
        "graph.html",
        train_data=data,
        predict_data=prediction_results
    )


# ---------------- DOWNLOAD MODEL ----------------
@app.route('/download')
def download():
    if 'user' not in session:
        return redirect('/')

    try:
        return send_file(MODEL_FILE, as_attachment=True)
    except:
        return "❌ Model not found"


# ---------------- LOGOUT ----------------
@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect('/')


# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)