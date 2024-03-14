import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Load data from the JSON lines files
eventos = pd.read_json("../../simulation.jsonlines", lines=True)
planes = pd.read_json("../../plans.jsonlines", lines=True)

# Process truck routes from plans
camiones = []
for simId in planes.simulationId.unique():
    for truck in planes[planes.simulationId == simId].trucks.values[0]:
        camiones.append(pd.DataFrame(truck["route"]).assign(simulationId=simId, truckId=truck["truck_id"]))
camiones = pd.concat(camiones)

# Calculate planned durations
tiempos_plan = camiones.sort_values(["simulationId", "truckId"]).assign(duration=lambda x: x["duration"] * 1000).groupby(["simulationId", "truckId"]).duration.agg(list).reset_index()
tiempos_plan.rename(columns={"duration": "tiempo_plan"}, inplace=True)

# Process event data
eventos = eventos.sort_values(["simulationId", "truckId", "eventTime"])
eventos["prev_event"] = eventos.groupby(["truckId", "simulationId"])["eventType"].shift(1)
eventos["prev_time"] = eventos.groupby(["truckId", "simulationId"])["eventTime"].shift(1)
eventos["delta"] = eventos.eventTime - eventos.prev_time
tiempos_sim = eventos[eventos.eventType.isin(["Truck arrived", "Truck ended route"])].sort_values(["simulationId", "truckId", "eventTime"]).groupby(["simulationId", "truckId"]).delta.agg(list).reset_index()
tiempos_sim.rename(columns={"delta": "tiempo_sim"}, inplace=True)
retrasos = tiempos_sim.merge(tiempos_plan, on=["simulationId", "truckId"]).dropna().reset_index(drop=True)

# Prepare data for model training
arr = np.array(retrasos.apply(lambda x: list(zip(x.tiempo_plan, x.tiempo_sim)), axis=1).explode())
arr = np.array(arr.tolist())
x = arr[:, 0].reshape(-1, 1) / 1000  # Planned time
y = arr[:, 1].reshape(-1, 1) / 1000  # Actual time

# Split data for travel model
x_train, x_test, y_train, y_test = train_test_split(x, y.ravel(), test_size=0.2, random_state=42)

# Train the travel model with RandomForestRegressor
travel_model = RandomForestRegressor(n_estimators=100, random_state=42)
travel_model.fit(x_train, y_train)

# Make predictions with the travel model
y_pred_travel = travel_model.predict(x_test)

# Calculate and print metrics for the travel model
print("Travel Model Metrics:")
print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_pred_travel)}")
print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred_travel)}")
print(f"R-squared (R²): {r2_score(y_test, y_pred_travel)}")

# Plot actual vs predicted values for the travel model
plt.figure(figsize=(10, 6))
plt.scatter(x_test, y_test, color='blue', label='Actual')
plt.scatter(x_test, y_pred_travel, color='red', alpha=0.5, label='Predicted')
plt.title('Travel Model: Actual vs Predicted')
plt.xlabel('Planned Time')
plt.ylabel('Actual Time')
plt.legend()
plt.show()

# Prepare data for the delivery model
tiemposEntrega = eventos[eventos.eventType == "Truck ended delivering"][["truckId", "delta"]]

# Label encoding for truckId
le = LabelEncoder()
tiemposEntrega["truckId"] = le.fit_transform(tiemposEntrega["truckId"])

# Prepare data for delivery model
X_deliv = tiemposEntrega["truckId"].values.reshape(-1, 1)
y_deliv = tiemposEntrega["delta"].values.ravel()
X_train_deliv, X_test_deliv, y_train_deliv, y_test_deliv = train_test_split(X_deliv, y_deliv, test_size=0.2, random_state=42)

# Train the delivery model with RandomForestRegressor
delivery_model = RandomForestRegressor(n_estimators=100, random_state=42)
delivery_model.fit(X_train_deliv, y_train_deliv)

# Make predictions with the delivery model
y_pred_delivery = delivery_model.predict(X_test_deliv)

# Calculate and print metrics for the delivery model
print("\nDelivery Model Metrics:")
print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test_deliv, y_pred_delivery)}")
print(f"Mean Squared Error (MSE): {mean_squared_error(y_test_deliv, y_pred_delivery)}")
print(f"R-squared (R²): {r2_score(y_test_deliv, y_pred_delivery)}")

# Plot actual vs predicted values for the delivery model
plt.figure(figsize=(10, 6))
plt.scatter(X_test_deliv, y_test_deliv, color='blue', label='Actual')
plt.scatter(X_test_deliv, y_pred_delivery, color='red', alpha=0.5, label='Predicted')
plt.title('Delivery Model: Actual vs Predicted')
plt.xlabel('Truck ID')
plt.ylabel('Delivery Time')
plt.legend()
plt.show()

# Optionally, save your models and the label encoder to disk
with open('travelModel.pkl', 'wb') as f:
    pickle.dump(travel_model, f)
with open('deliveryModel.pkl', 'wb') as f:
    pickle.dump(delivery_model, f)
with open('le.pkl', 'wb') as f:
    pickle.dump(le, f)