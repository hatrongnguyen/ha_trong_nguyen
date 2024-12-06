import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
import joblib
import mlflow
import mlflow.sklearn
import numpy as np

def generate_data(output_file="generated_data.csv"):
    """
    Tạo dữ liệu phân loại giả lập và trả về một DataFrame. 
    Dữ liệu được lưu vào file CSV để kiểm tra.

    Args:
        output_file (str): Đường dẫn file CSV để lưu dữ liệu.

    Returns:
        data (pd.DataFrame): DataFrame chứa dữ liệu giả lập
    """
    # Tạo dữ liệu giả lập
    X, y = make_classification(
        n_samples=1000,       # Số lượng mẫu
        n_features=5,         # Số lượng đặc trưng (5 thay vì 4 để đảm bảo có đủ cột dữ liệu)
        n_informative=4,      # Số đặc trưng quan trọng
        n_redundant=0,        # Số đặc trưng dư thừa
        n_classes=2,          # Phân loại nhị phân
        random_state=42
    )
    X = np.abs(X)  # Loại bỏ giá trị âm trong dữ liệu

    # Đặt tên cho các đặc trưng
    columns = ['Age', 'Income', 'Spending Score', 'Previous Purchases', 'Days Since Last Purchase']
    data = pd.DataFrame(X, columns=columns)

    # Thêm nhãn mục tiêu vào DataFrame
    data['Target'] = y

    # Thêm một số giá trị thiếu để mô phỏng dữ liệu thực tế
    data.loc[data.sample(frac=0.1, random_state=42).index, 'Income'] = None

    # Lưu dữ liệu ra file CSV
    data.to_csv(output_file, index=False)
    print(f"Dữ liệu đã được lưu vào file: {output_file}")

    return data

# 2. Hàm tiền xử lý dữ liệu
def preprocess_data(data):
    """
    Tiền xử lý dữ liệu:
    - Xử lý giá trị thiếu
    - Chuẩn hóa dữ liệu số

    Args:
        data (pd.DataFrame): Dữ liệu đầu vào

    Returns:
        X (pd.DataFrame): Dữ liệu đầu vào đã tiền xử lý
        y (pd.Series): Nhãn mục tiêu
    """
    print("Bắt đầu tiền xử lý dữ liệu...")

    # Tách nhãn mục tiêu
    y = data['Target']
    X = data.drop(columns=['Target'])

    # Xử lý giá trị thiếu
    imputer = SimpleImputer(strategy='mean')
    X[['Income']] = imputer.fit_transform(X[['Income']])

    # Chuẩn hóa các đặc trưng số
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)

    print("Hoàn thành tiền xử lý dữ liệu.")
    return X, y

# 3. Hàm huấn luyện mô hình
def train_model(X_train, y_train):
    """
    Huấn luyện mô hình phân loại sử dụng Random Forest với GridSearchCV.

    Args:
        X_train (pd.DataFrame): Dữ liệu huấn luyện.
        y_train (pd.Series): Nhãn của dữ liệu huấn luyện.

    Returns:
        best_model (RandomForestClassifier): Mô hình tốt nhất sau khi tuning.
    """
    model = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [10, 50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    print("Tuning hoàn thành. Tìm thấy mô hình tốt nhất.")
    return grid_search.best_estimator_

# 4. Hàm đánh giá mô hình
def evaluate_model(model, X_test, y_test):
    """
    Đánh giá mô hình và in độ chính xác.

    Args:
        model: Mô hình đã được huấn luyện.
        X_test (pd.DataFrame): Dữ liệu kiểm tra.
        y_test (pd.Series): Nhãn của dữ liệu kiểm tra.

    Returns:
        accuracy (float): Độ chính xác của mô hình trên dữ liệu kiểm tra.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy of the model: {accuracy:.2f}")
    return accuracy

# 5. Pipeline chính
def main_pipeline():
    """
    Pipeline chính để thực thi toàn bộ luồng công việc:
    - Tạo dữ liệu
    - Tiền xử lý dữ liệu
    - Chia dữ liệu
    - Huấn luyện mô hình
    - Đánh giá mô hình
    - Lưu mô hình tốt nhất
    """
    print("Bắt đầu pipeline...")
    # Bước 1: Tạo dữ liệu
    data = generate_data()

    # Bước 2: Tiền xử lý dữ liệu
    X, y = preprocess_data(data)

    # Bước 3: Chia tập dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print("Dữ liệu đã được chia thành tập huấn luyện và kiểm tra.")

    mlflow.set_experiment("customer_purchase_prediction")
    with mlflow.start_run() as run:
        # Huấn luyện mô hình
        best_model = train_model(X_train, y_train)

        # Đánh giá mô hình
        accuracy = evaluate_model(best_model, X_test, y_test)

        # Log các siêu tham số và kết quả
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("test_size", 0.3)
        mlflow.log_metric("accuracy", accuracy)

        # Lưu mô hình vào MLflow
        mlflow.sklearn.log_model(best_model, "best_model")

        # Lấy đường dẫn nơi lưu mô hình
        artifact_uri = mlflow.get_artifact_uri("best_model")
        print(f"Model saved at: {artifact_uri}")

if __name__ == "__main__":
    main_pipeline()
