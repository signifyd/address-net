from addressnet.predict import predict_one

if __name__ == "__main__":
    model_dir = '/Users/niaschmald/tmp/tfmodel'
    print(predict_one("casa del gelato, 10A 24-26 high street road mount waverley vic 3183"), model_dir=model_dir)
