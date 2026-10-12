# my_experiment.py

# 1. 从我们提供的帮助脚本中导入加载函数
import pandas as pd

def load_data():
    train_df = pd.read_csv('train_data.csv')
    test_df = pd.read_csv('test_data_unlabeled.csv')
    X_train = train_df['text'].astype(str).tolist()
    y_train = train_df['target'].values
    X_test_unlabeled = test_df['text'].astype(str).tolist()
    return X_train, y_train, X_test_unlabeled

# 2. 调用函数来获取数据
#    这个函数会自动读取 .csv 文件并返回你需要的所有内容
X_train, y_train, X_test_unlabeled = load_data()

# 3. (验证步骤) 检查一下数据是否加载成功
print("--- 数据加载成功 ---")
print(f"训练集样本数量: {len(X_train)}")
print(f"训练集标签数量: {len(y_train)}")
print(f"无标签测试集样本数量: {len(X_test_unlabeled)}")
print("-" * 20)

# 打印第一个训练样本和它的标签，感受一下数据
print("第一个训练样本内容:")
print(X_train[0])
print(f"\n第一个训练样本的标签: {y_train[0]}")
print("-" * 20)

# 打印第一个需要你预测的测试样本
print("第一个无标签测试样本内容:")
print(X_test_unlabeled[0])
print("\n" + "="*50)

# --- 在这里开始你的实验！ ---
# 现在，你可以使用 X_train, y_train, 和 X_test_unlabeled 这三个变量
# 来进行后续的特征提取、模型训练和预测了。

# 举例：
# 1. 创建TF-IDF向量化器
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)

# 2. 训练一个模型...
from `sklearn.svm` import SVC
svm_model = SVC()
svm_model.fit(X_train_tfidf, y_train)

# 3. 对测试集进行预测...
X_test_tfidf = vectorizer.transform(X_test_unlabeled)
predictions = svm_model.predict(X_test_tfidf)

# 4. 保存你的预测结果...
import pandas as pd
pd.DataFrame(predictions).to_csv('predictions.csv', index=False, header=False)