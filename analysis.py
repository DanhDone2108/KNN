
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def train_model(x, y, ngh=3):
    #Tách dữ liệu của từng cột ra để train và test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    #khoi tao mo hinh
    model = KNeighborsClassifier(n_neighbors=ngh)
    #Huấn luyen mo hình
    model.fit(x_train, y_train)
    #Dự đoán Nhãn Cho Tập kiễm tra
    y_pred = model.predict(x_test)
    #Dánh gia mô hình sau khi kiểm tra
    accuracy = accuracy_score(y_test, y_pred)
    #in do chinh xac của thuạt toan phan loai
    print("Accuracy: {:.2f}%".format(accuracy * 100.0))
    report = (classification_report(y_test, y_pred))
    # luu mo hinh sau khi phan tich xong
    joblib.dump(model, 'knn_model.pkl')

    return accuracy, report
