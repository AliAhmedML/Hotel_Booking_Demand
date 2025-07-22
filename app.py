from flask import Flask, render_template, request
from predictor import HotelBookingPredictor

app = Flask(__name__)
predictor = HotelBookingPredictor()


@app.route("/", methods=["GET", "POST"])
def predict():
    prediction = None
    if request.method == "POST":
        prediction = predictor.predict(request.form)
    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
