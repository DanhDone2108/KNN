from os import access
from sklearn.metrics import accuracy_score
from manger import load_data, preprocess_data
from analysis import train_model

def main():
    data = load_data("/Users/danh/Downloads/File down /Dữ liệu KNN/MachineLearningCVE/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv")
    print(data.columns)

    try:
        x,y = preprocess_data(data)

        accuracy, report = train_model(x,y)
        print("Precision level",accuracy)
        print("Report:\n", report)
    except KeyError as e:
        print(f"KeyError: {e}")

if __name__ == '__main__':
    main()


