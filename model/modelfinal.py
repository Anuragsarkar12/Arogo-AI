import pandas as pd

def load_and_preprocess_data(file_path):
    data = pd.read_excel(file_path)
    data['Vehicle Type'].fillna('Unknown', inplace=True)
    data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_')
    data = data.drop(columns=['shipment_id'])
    return data



def encode_data(data):
    columns_to_encode = ['origin', 'destination', 'weather_conditions', 'traffic_conditions', 'vehicle_type',]
    data_encoded = pd.get_dummies(data, columns=columns_to_encode, drop_first=False)
    data_encoded['delayed'] = data_encoded['delayed'].map({'Yes': 1, 'No': 0})
    return data_encoded

def preprocess_dates(data_encoded):
    data_encoded = data_encoded.drop(columns=['actual_delivery_date'])
    data_encoded['shipment_date'] = data_encoded['shipment_date'].view('int64') / 10**9  
    data_encoded['planned_delivery_date'] = data_encoded['planned_delivery_date'].view('int64') / 10**9
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    data_encoded[['shipment_date', 'planned_delivery_date']] = scaler.fit_transform(data_encoded[['shipment_date', 'planned_delivery_date']])
    return data_encoded, scaler

def train_and_evaluate_models(X_train, X_test, y_train, y_test, scaler):
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    import joblib

    model1 = LogisticRegression(max_iter=5000)
    model2 = DecisionTreeClassifier()
    model3 = RandomForestClassifier()

    model1.fit(X_train, y_train)
    y_pred1 = model1.predict(X_test)
    print("Logistic Regression Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred1):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred1, average='binary'):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred1, average='binary'):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred1, average='binary'):.4f}")
    print("-" * 40)
    joblib.dump(model1, 'model/logistic_regression_model_final.pkl')
    joblib.dump(scaler, 'model/scaler_final.pkl')  

    model2.fit(X_train, y_train)
    y_pred2 = model2.predict(X_test)
    print("Decision Tree Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred2):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred2, average='binary'):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred2, average='binary'):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred2, average='binary'):.4f}")
    print("-" * 40)
    joblib.dump(model2, 'model/decision_tree_model_final.pkl')

    model3.fit(X_train, y_train)
    y_pred3 = model3.predict(X_test)
    print("Random Forest Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred3):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred3, average='binary'):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred3, average='binary'):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred3, average='binary'):.4f}")
    print("-" * 40)
    joblib.dump(model3, 'model/random_forest_model_final.pkl')

def main():
    file_path = 'AI ML Internship Training Data.xlsx'
    data = load_and_preprocess_data(file_path)
    data_encoded = encode_data(data)
    data_encoded, scaler = preprocess_dates(data_encoded)
    X = data_encoded.drop(columns=['delayed'])
    y = data_encoded['delayed']
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_and_evaluate_models(X_train, X_test, y_train, y_test, scaler)

if __name__ == "__main__":
    main()
