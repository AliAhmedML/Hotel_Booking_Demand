import pickle
import pandas as pd
import cudf


class HotelBookingPredictor:
    def __init__(self):
        self.model = pickle.load(open("Trained_models/Hotel.pkl", "rb"))
        self.encoder = pickle.load(open("Trained_models/One_Hot_Encoder.pkl", "rb"))
        self.pt = pickle.load(open("Trained_models/Power_Transformer.pkl", "rb"))
        self.scaler = pickle.load(open("Trained_models/Scaler.pkl", "rb"))
        self.input_features = [
            "Booking_ID",
            "number_of_adults",
            "number_of_children",
            "number_of_weekend_nights",
            "number_of_week_nights",
            "type_of_meal",
            "car_parking_space",
            "room_type",
            "lead_time",
            "market_segment_type",
            "repeated",
            "P_C",
            "P_not_C",
            "average_price_",
            "special_requests",
            "month",
            "day",
            "year",
        ]
        self.cat_cols = ["type_of_meal", "room_type", "market_segment_type"]
        self.pt_cols = [
            "number_of_weekend_nights",
            "number_of_week_nights",
            "lead_time",
            "special_requests",
        ]

    def preprocess(self, form):
        data = {
            feature: form[feature]
            for feature in self.input_features
            if feature != "Booking_ID"
        }
        df = pd.DataFrame([data])

        # One-hot encode categorical columns
        encoded = self.encoder.transform(df[self.cat_cols])
        encoded_df = pd.DataFrame(
            encoded,
            columns=self.encoder.get_feature_names_out(self.cat_cols),
            index=df.index,
        )
        df = pd.concat([df.drop(self.cat_cols, axis=1), encoded_df], axis=1)

        # Power transform
        df[self.pt_cols] = self.pt.transform(df[self.pt_cols])

        # Scale
        df = self.scaler.transform(df)
        df = cudf.DataFrame(df).astype("float32")
        return df

    def predict(self, form):
        df = self.preprocess(form)
        prediction = self.model.predict(df)
        return "Not Canceled" if prediction[0] == 1 else "Canceled"
