import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


RANDOM_STATE = 42
TARGET = "Price($)"

st.set_page_config(
    page_title="Used Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)


@st.cache_data
def load_data(file_path="car_price_dataset.csv"):
    return pd.read_csv(file_path)


@st.cache_resource
def train_models(file_path="car_price_dataset.csv"):
    df = load_data(file_path).copy()

    # Same cleaning used in the notebook
    df["PricePerKm"] = df["PricePerKm"].fillna(df["PricePerKm"].mean())

    categorical_cols = [
        "Brand", "Condition", "FuelType", "Transmission", "DriveType",
        "BodyType", "Doors", "Seats", "Color", "Interior", "City",
        "AccidentHistory", "Insurance", "RegistrationStatus"
    ]

    df[categorical_cols] = df[categorical_cols].astype("category")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            min_frequency=5
        ))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    # OLS
    ols_model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ])

    ols_model.fit(X_train, y_train)
    ols_pred = ols_model.predict(X_test)

    # SGD
    sgd_model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", SGDRegressor(
            loss="squared_error",
            penalty="l2",
            max_iter=2000,
            tol=1e-3,
            random_state=RANDOM_STATE
        ))
    ])

    sgd_model.fit(X_train, y_train)
    sgd_pred = sgd_model.predict(X_test)

    # Polynomial Regression
    polynomial_results = []
    polynomial_models = {}

    for degree in [2, 3]:
        polynomial_numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("polynomial", PolynomialFeatures(
                degree=degree,
                include_bias=False
            )),
            ("scaler", StandardScaler())
        ])

        polynomial_preprocessor = ColumnTransformer([
            ("num", polynomial_numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features)
        ])

        model = Pipeline([
            ("preprocessor", polynomial_preprocessor),
            ("model", LinearRegression())
        ])

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mse = mean_squared_error(y_test, predictions)

        polynomial_results.append({
            "Model": f"Polynomial Regression (Degree {degree})",
            "MAE": mean_absolute_error(y_test, predictions),
            "MSE": mse,
            "RMSE": np.sqrt(mse),
            "R²": r2_score(y_test, predictions)
        })

        polynomial_models[degree] = model

    best_degree = min(
        [2, 3],
        key=lambda d: polynomial_results[[2, 3].index(d)]["RMSE"]
    )

    models = {
        "OLS": ols_model,
        "SGD": sgd_model,
        "Polynomial Regression (Degree 2)": polynomial_models[2],
        "Polynomial Regression (Degree 3)": polynomial_models[3]
    }

    predictions_map = {
        "OLS": ols_pred,
        "SGD": sgd_pred,
        "Polynomial Regression (Degree 2)": polynomial_models[2].predict(X_test),
        "Polynomial Regression (Degree 3)": polynomial_models[3].predict(X_test)
    }

    results = []

    for name, predictions in predictions_map.items():
        mse = mean_squared_error(y_test, predictions)
        results.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, predictions),
            "MSE": mse,
            "RMSE": np.sqrt(mse),
            "R²": r2_score(y_test, predictions)
        })

    results_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)

    return {
        "df": df,
        "models": models,
        "results": results_df,
        "best_model": results_df.loc[0, "Model"],
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "X_test": X_test,
        "y_test": y_test
    }


def format_money(value):
    return f"${value:,.2f}"


st.title("🚗 Used Car Price Prediction")
st.write(
    "Predict the selling price of a used car using the regression workflow "
    "from the project notebook."
)

st.sidebar.header("⚙️ Model Settings")

data_file = st.sidebar.text_input(
    "Dataset path",
    value="car_price_dataset.csv"
)

try:
    with st.spinner("Loading dataset and training regression models..."):
        project = train_models(data_file)
except FileNotFoundError:
    st.error(
        f"Dataset not found: `{data_file}`. "
        "Put `car_price_dataset.csv` in the same folder as `app.py`."
    )
    st.stop()
except Exception as e:
    st.error(f"Could not train the models: {e}")
    st.stop()


df = project["df"]
models = project["models"]
results_df = project["results"]

st.sidebar.success(f"Dataset: {len(df):,} rows")

best_model = results_df.loc[0, "Model"]

st.sidebar.markdown("### Best Model")
st.sidebar.info(best_model)

tab1, tab2, tab3 = st.tabs([
    "🚗 Prediction",
    "📊 Model Performance",
    "📋 Dataset"
])


with tab1:
    st.subheader("Enter Car Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        brand = st.selectbox("Brand", sorted(df["Brand"].dropna().unique()))
        model = st.selectbox("Model", sorted(df["Model"].dropna().unique()))
        year = st.number_input(
            "Year",
            min_value=int(df["Year"].min()),
            max_value=int(df["Year"].max()),
            value=int(df["Year"].median()),
            step=1
        )
        car_age = st.number_input(
            "Car Age",
            min_value=0,
            max_value=100,
            value=max(0, 2026 - int(year)),
            step=1
        )
        condition = st.selectbox(
            "Condition",
            sorted(df["Condition"].dropna().unique())
        )
        mileage = st.number_input(
            "Mileage (km)",
            min_value=0,
            value=int(df["Mileage(km)"].median()),
            step=1000
        )
        engine_size = st.number_input(
            "Engine Size (L)",
            min_value=float(df["EngineSize(L)"].min()),
            max_value=float(df["EngineSize(L)"].max()),
            value=float(df["EngineSize(L)"].median()),
            step=0.1
        )
        fuel_type = st.selectbox(
            "Fuel Type",
            sorted(df["FuelType"].dropna().unique())
        )

    with col2:
        horsepower = st.number_input(
            "Horsepower",
            min_value=int(df["Horsepower"].min()),
            max_value=int(df["Horsepower"].max()),
            value=int(df["Horsepower"].median()),
            step=1
        )
        torque = st.number_input(
            "Torque",
            min_value=int(df["Torque"].min()),
            max_value=int(df["Torque"].max()),
            value=int(df["Torque"].median()),
            step=1
        )
        transmission = st.selectbox(
            "Transmission",
            sorted(df["Transmission"].dropna().unique())
        )
        drive_type = st.selectbox(
            "Drive Type",
            sorted(df["DriveType"].dropna().unique())
        )
        body_type = st.selectbox(
            "Body Type",
            sorted(df["BodyType"].dropna().unique())
        )
        doors = st.selectbox(
            "Doors",
            sorted(df["Doors"].dropna().unique())
        )
        seats = st.selectbox(
            "Seats",
            sorted(df["Seats"].dropna().unique())
        )
        color = st.selectbox(
            "Color",
            sorted(df["Color"].dropna().unique())
        )

    with col3:
        interior = st.selectbox(
            "Interior",
            sorted(df["Interior"].dropna().unique())
        )
        options = st.selectbox(
            "Options",
            sorted(df["Options"].dropna().unique())
        )
        city = st.selectbox(
            "City",
            sorted(df["City"].dropna().unique())
        )
        accident_history = st.selectbox(
            "Accident History",
            sorted(df["AccidentHistory"].dropna().unique())
        )
        insurance = st.selectbox(
            "Insurance",
            sorted(df["Insurance"].dropna().unique())
        )
        registration_status = st.selectbox(
            "Registration Status",
            sorted(df["RegistrationStatus"].dropna().unique())
        )
        fuel_efficiency = st.number_input(
            "Fuel Efficiency (L/100km)",
            min_value=float(df["FuelEfficiency(L/100km)"].min()),
            max_value=float(df["FuelEfficiency(L/100km)"].max()),
            value=float(df["FuelEfficiency(L/100km)"].median()),
            step=0.1
        )
        price_per_km = st.number_input(
            "Price Per Km",
            min_value=float(df["PricePerKm"].min()),
            max_value=float(df["PricePerKm"].max()),
            value=float(df["PricePerKm"].median()),
            step=0.01
        )

    st.divider()

    selected_model = st.selectbox(
        "Choose Regression Model",
        list(models.keys()),
        index=list(models.keys()).index(best_model)
    )

    if st.button("🔮 Predict Car Price", type="primary", use_container_width=True):
        input_data = pd.DataFrame([{
            "Brand": brand,
            "Model": model,
            "Year": year,
            "CarAge": car_age,
            "Condition": condition,
            "Mileage(km)": mileage,
            "EngineSize(L)": engine_size,
            "FuelType": fuel_type,
            "Horsepower": horsepower,
            "Torque": torque,
            "Transmission": transmission,
            "DriveType": drive_type,
            "BodyType": body_type,
            "Doors": doors,
            "Seats": seats,
            "Color": color,
            "Interior": interior,
            "Options": options,
            "City": city,
            "AccidentHistory": accident_history,
            "Insurance": insurance,
            "RegistrationStatus": registration_status,
            "FuelEfficiency(L/100km)": fuel_efficiency,
            "PricePerKm": price_per_km
        }])

        prediction = models[selected_model].predict(input_data)[0]

        st.success(f"Estimated Car Price: {format_money(prediction)}")
        st.metric(
            "Predicted Price",
            format_money(prediction)
        )


with tab2:
    st.subheader("Regression Model Comparison")

    display_results = results_df.copy()
    display_results["MAE"] = display_results["MAE"].map(
        lambda x: f"${x:,.2f}"
    )
    display_results["MSE"] = display_results["MSE"].map(
        lambda x: f"{x:,.2f}"
    )
    display_results["RMSE"] = display_results["RMSE"].map(
        lambda x: f"${x:,.2f}"
    )
    display_results["R²"] = display_results["R²"].map(
        lambda x: f"{x:.4f}"
    )

    st.dataframe(
        display_results,
        use_container_width=True,
        hide_index=True
    )

    st.markdown(f"### 🏆 Best Model: `{best_model}`")
    st.write(
        "The best model is selected according to the lowest RMSE on the "
        "test set, following the notebook workflow."
    )


with tab3:
    st.subheader("Dataset Overview")

    c1, c2, c3 = st.columns(3)

    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Features", f"{df.shape[1] - 1}")
    c3.metric("Target", TARGET)

    st.dataframe(
        df.head(100),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Feature Information")
    st.write("Numerical Features:")
    st.code(", ".join(project["numeric_features"]))

    st.write("Categorical Features:")
    st.code(", ".join(project["categorical_features"]))


st.divider()
st.caption(
    "Used Car Price Prediction • Scikit-learn + Streamlit • "
    "Based on the supplied Linear Regression project notebook"
)
