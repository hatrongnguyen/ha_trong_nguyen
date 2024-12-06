from flask import Flask, request, jsonify, render_template
import mlflow
import mlflow.sklearn

# Tạo ứng dụng Flask
app = Flask(__name__)

# Route trang chủ, trả về form để nhập dữ liệu
@app.route('/')
def home():     
    return render_template('fill.html') 

# Route dự đoán khả năng mua hàng
@app.route("/predict/", methods=["POST"])
def predict():
    try:
        # Lấy dữ liệu từ form người dùng
        age = int(request.form['age'])  # Độ tuổi của khách hàng
        income = float(request.form['income'])  # Thu nhập của khách hàng
        spending_score = float(request.form['spending_score'])  # Điểm chi tiêu của khách hàng
        previous_purchases = int(request.form['previous_purchases'])  # Số lần mua hàng trước đó
        days_since_last_purchase = float(request.form['days_since_last_purchase'])  # Số ngày từ lần mua hàng gần nhất

        # Dữ liệu đầu vào cho mô hình
        input_data = [[age, income, spending_score, previous_purchases, days_since_last_purchase]]

        # Load mô hình đã được lưu từ MLflow
        model_uri = "file:///D:/code/python/MSE/MLOps/prj_mlops_htn/mlruns/536343250310809735/c5a030e03b744d578a75464ae8914923/artifacts/best_model"  # Cập nhật với URI đúng
        model = mlflow.sklearn.load_model(model_uri)
        
        # Dự đoán
        submit = model.predict(input_data).tolist()

        # Trả về kết quả dự đoán
        return jsonify({"submit": submit[0]})
    
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
