import joblib
import pandas as pd
import numpy as np

# Load the trained model pipeline
model = joblib.load("random_forest_model.pkl")


def get_user_input(simple=True):
    if simple:
        print("Simple Mode (fewer inputs, may be less accurate)\n")
    else:
        print("Advanced Mode (more inputs, more accurate)\n")

    prod_year = int(input("Year of Manufacture (e.g., 2018): "))
    mileage = float(input("Mileage (in km): "))
    engine_volume = float(input("Engine Volume (in L): "))
    car_age = 2025 - prod_year

    if simple:
        fuel_type = input(
            "Fuel Type (Petrol/Diesel/Gas/Electric/Hybrid): ").strip(
                ).capitalize()
        return {
            "Prod. year": prod_year,
            "Car_Age": car_age,
            "Levy": 0.0,
            "Manufacturer": "Unknown",
            "Color": "Black",
            "Category": "Sedan",
            "Wheel": "Left",
            "Gear box type": "Automatic",
            "Leather interior": 0,
            "Fuel type": fuel_type,
            "Transmission": "Automatic",
            "Drive wheels": "Front",
            "Doors": 4,
            "Airbags": 2,
            "Mileage": mileage,
            "Engine volume": engine_volume,
            "Cylinders": 4
        }

    # Advanced mode
    levy = float(input("Levy (enter 0 if unknown): "))
    manufacturer = input("Manufacturer (e.g., BMW, Toyota): ").strip()
    color = input("Color (e.g., Black, White): ").strip()
    category = input("Category (e.g., Jeep, Sedan): ").strip()
    wheel = input("Wheel (Left/Right): ").strip().capitalize()
    gearbox_type = input(
        "Gear box type (Automatic/Manual): ").strip().capitalize()
    leather_interior = input(
        "Leather Interior? (Yes/No): ").strip().capitalize()
    fuel_type = input(
        "Fuel Type (Petrol/Diesel/Gas/Electric/Hybrid): ").strip().capitalize()
    transmission = input
    ("Transmission (Automatic/Manual): ").strip().capitalize()
    drive_wheels = input(
        "Drive Wheels (Front/Rear/4x4): ").strip().capitalize()
    doors = int(input("Number of Doors: "))
    airbags = int(input("Number of Airbags (2, 4, etc.): "))
    cylinders = int(input("Number of Cylinders: "))

    return {
        "Prod. year": prod_year,
        "Car_Age": car_age,
        "Levy": levy,
        "Manufacturer": manufacturer,
        "Color": color,
        "Category": category,
        "Wheel": wheel,
        "Gear box type": gearbox_type,
        "Leather interior": 1 if leather_interior == "Yes" else 0,
        "Fuel type": fuel_type,
        "Transmission": transmission,
        "Drive wheels": drive_wheels,
        "Doors": doors,
        "Airbags": airbags,
        "Mileage": mileage,
        "Engine volume": engine_volume,
        "Cylinders": cylinders
    }


def make_prediction(car_details):
    df = pd.DataFrame([car_details])
    log_price = model.predict(df)[0]
    return np.expm1(log_price)


def main():
    while True:
        print("Car Price Prediction CLI: \n")
        try:
            # Ask for the mode right at the beginning
            mode = input(
                "Mode: Would you like Simple or Advanced? (s/a): ").strip(
                    ).lower()
            if mode == "s":
                simple_mode = True
            elif mode == "a":
                simple_mode = False
            else:
                print("Invalid option. Please choose 's' or 'a'")
                continue

            car_details = get_user_input(simple=simple_mode)
            predicted_price = make_prediction(car_details)
            print(f"\nEstimated Car Price: ${predicted_price:,.2f}\n")
            break
        except Exception as e:
            print("\nError has occured:")
            print(str(e))
            retry = input("\nTry again? (y/n): ").strip().lower()
            if retry != "y":
                print("Exiting program.")
                break


if __name__ == "__main__":
    main()
