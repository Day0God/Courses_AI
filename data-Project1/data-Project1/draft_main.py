# my_experiment.py

# 1. 从我们提供的帮助脚本中导入加载函数
import pandas as pd
import time

def load_data():
    train_df = pd.read_csv('train_data.csv')
    test_df = pd.read_csv('test_data_unlabeled.csv')
    X_train = train_df['text'].astype(str).tolist()
    y_train = train_df['target'].values
    X_test_unlabeled = test_df['text'].astype(str).tolist()
    return X_train, y_train, X_test_unlabeled

# 2. 调用函数来获取数据
X_train, y_train, X_test_unlabeled = load_data()

# 3. (验证步骤) 检查一下数据是否加载成功
print("--- 数据加载成功 ---")
print(f"训练集样本数量: {len(X_train)}")
print(f"训练集标签数量: {len(y_train)}")
print(f"无标签测试集样本数量: {len(X_test_unlabeled)}")
print("-" * 20)

print("第一个训练样本内容:")
print(X_train[0])
print(f"\n第一个训练样本的标签: {y_train[0]}")
print("-" * 20)

print("第一个无标签测试样本内容:")
print(X_test_unlabeled[0])
print("\n" + "="*50)

# ============ 基线实验开始 ============

import numpy as np
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

from sklearn.model_selection import train_test_split
X_train_part, X_val, y_train_part, y_val = train_test_split(
    X_train, y_train,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y_train
)
print("--- 验证集划分完成 ---")
print(f"训练子集: {len(X_train_part)} | 验证集: {len(X_val)}")

# TF-IDF 向量化（特征工程：10000维 + bigram + 去停用词）
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    stop_words='english'
)
X_train_vec = vectorizer.fit_transform(X_train_part)
X_val_vec   = vectorizer.transform(X_val)

print(f"TF-IDF 特征维度: {X_train_vec.shape[1]}")

from sklearn.metrics import accuracy_score, f1_score, classification_report

def evaluate_model(model, X_train_data, X_val_data, y_train_data, y_val_data, model_name="模型"):
    """在训练集和验证集上评估模型，打印完整指标，辅助判断欠/过拟合"""
    # 训练集评估
    y_train_pred = model.predict(X_train_data)
    train_acc = accuracy_score(y_train_data, y_train_pred)
    train_f1_macro = f1_score(y_train_data, y_train_pred, average='macro')
    
    # 验证集评估
    y_val_pred = model.predict(X_val_data)
    val_acc = accuracy_score(y_val_data, y_val_pred)
    val_f1_macro = f1_score(y_val_data, y_val_pred, average='macro')
    val_f1_micro = f1_score(y_val_data, y_val_pred, average='micro')
    val_f1_weighted = f1_score(y_val_data, y_val_pred, average='weighted')
    
    print(f"\n--- {model_name} 最佳参数 训练集 vs 验证集 ---")
    print(f"训练集 → Accuracy: {train_acc:.4f}, F1-macro: {train_f1_macro:.4f}")
    print(f"验证集 → Accuracy: {val_acc:.4f}, F1-macro: {val_f1_macro:.4f}")
    print(f"差距   → Acc差距: {train_acc - val_acc:.4f}, F1差距: {train_f1_macro - val_f1_macro:.4f}")
    
    if train_acc - val_acc > 0.05:
        print("⚠️  训练集明显优于验证集 → 可能存在过拟合")
    elif train_acc < 0.85:
        print("⚠️  训练集表现就很低 → 可能存在欠拟合")
    else:
        print("✅ 训练/验证差距合理，泛化良好")
    
    print(f"\n验证集详细指标:")
    print(f"  F1-micro:    {val_f1_micro:.4f}")
    print(f"  F1-weighted: {val_f1_weighted:.4f}")
    print("\n分类报告:")
    print(classification_report(y_val_data, y_val_pred, digits=4))
    
    return {
        'model_name': model_name,
        'model': model,
        'train_acc': train_acc,
        'train_f1_macro': train_f1_macro,
        'val_acc': val_acc,
        'val_f1_macro': val_f1_macro,
        'val_f1_micro': val_f1_micro,
        'val_f1_weighted': val_f1_weighted,
        'y_val_pred': y_val_pred
    }

# ============ 算法一：朴素贝叶斯 ============
from sklearn.naive_bayes import MultinomialNB

alpha_candidates = [0.01, 0.1, 0.5, 1.0]

best_alpha = None
best_val_acc = 0
best_val_f1_macro = 0
best_nb_model = None
nb_tuning_results = []

print("\n" + "=" * 60)
print("--- 算法一：朴素贝叶斯 (MultinomialNB) 超参数调优 ---")
print("=" * 60)
print("\n快速调参结果（alpha）:")
for alpha in alpha_candidates:
    t_start = time.time()
    nb = MultinomialNB(alpha=alpha)
    nb.fit(X_train_vec, y_train_part)
    t_end = time.time()
    t_elapsed = t_end - t_start
    
    y_val_pred = nb.predict(X_val_vec)
    val_acc = accuracy_score(y_val, y_val_pred)
    val_f1_macro = f1_score(y_val, y_val_pred, average='macro')

    nb_tuning_results.append({
        'alpha': alpha,
        'val_acc': val_acc,
        'val_f1_macro': val_f1_macro,
        'time_sec': t_elapsed
    })

    print(f"  alpha={alpha:.2f} → val_acc={val_acc:.4f}, val_f1_macro={val_f1_macro:.4f} (耗时: {t_elapsed:.2f}s)")

    if val_f1_macro > best_val_f1_macro:
        best_val_f1_macro = val_f1_macro
        best_val_acc = val_acc
        best_alpha = alpha
        best_nb_model = nb

print(f"\n最佳 alpha: {best_alpha}")
print(f"最佳验证集准确率: {best_val_acc:.4f}")
print(f"最佳验证集 F1-macro: {best_val_f1_macro:.4f}")

nb_best_result = evaluate_model(best_nb_model, X_train_vec, X_val_vec, y_train_part, y_val, "朴素贝叶斯 (alpha={})".format(best_alpha))

# ============ 算法二：支持向量机 (LinearSVC) ============
from sklearn.svm import LinearSVC

# C 值缩小范围，压制过拟合
svm_C_candidates = [0.001, 0.01, 0.1, 1.0]

best_svm_C = None
best_svm_val_acc = 0
best_svm_val_f1_macro = 0
best_svm_model = None
svm_tuning_results = []

print("\n" + "=" * 60)
print("--- 算法二：支持向量机 (LinearSVC) 超参数调优 ---")
print("=" * 60)
print("\n快速调参结果（C）:")
for C in svm_C_candidates:
    t_start = time.time()
    svm = LinearSVC(C=C, random_state=RANDOM_STATE, max_iter=1000)
    svm.fit(X_train_vec, y_train_part)
    t_end = time.time()
    t_elapsed = t_end - t_start
    
    y_val_pred = svm.predict(X_val_vec)
    val_acc = accuracy_score(y_val, y_val_pred)
    val_f1_macro = f1_score(y_val, y_val_pred, average='macro')

    svm_tuning_results.append({
        'C': C,
        'val_acc': val_acc,
        'val_f1_macro': val_f1_macro,
        'time_sec': t_elapsed
    })

    print(f"  C={C} → val_acc={val_acc:.4f}, val_f1_macro={val_f1_macro:.4f} (耗时: {t_elapsed:.2f}s)")

    if val_f1_macro > best_svm_val_f1_macro:
        best_svm_val_f1_macro = val_f1_macro
        best_svm_val_acc = val_acc
        best_svm_C = C
        best_svm_model = svm

print(f"\n最佳 C: {best_svm_C}")
print(f"最佳验证集准确率: {best_svm_val_acc:.4f}")
print(f"最佳验证集 F1-macro: {best_svm_val_f1_macro:.4f}")

svm_best_result = evaluate_model(best_svm_model, X_train_vec, X_val_vec, y_train_part, y_val, "LinearSVC (C={})".format(best_svm_C))

# ============ 算法三：逻辑回归 ============
from sklearn.linear_model import LogisticRegression

# C 值缩小范围，压制过拟合
lr_C_candidates = [0.001, 0.01, 0.1, 1.0]

best_lr_C = None
best_lr_val_acc = 0
best_lr_val_f1_macro = 0
best_lr_model = None
lr_tuning_results = []

print("\n" + "=" * 60)
print("--- 算法三：逻辑回归 (LogisticRegression) 超参数调优 ---")
print("=" * 60)
print("\n快速调参结果（C）:")
for C in lr_C_candidates:
    t_start = time.time()
    lr = LogisticRegression(C=C, max_iter=2000, random_state=RANDOM_STATE)
    lr.fit(X_train_vec, y_train_part)
    t_end = time.time()
    t_elapsed = t_end - t_start
    
    y_val_pred = lr.predict(X_val_vec)
    val_acc = accuracy_score(y_val, y_val_pred)
    val_f1_macro = f1_score(y_val, y_val_pred, average='macro')

    lr_tuning_results.append({
        'C': C,
        'val_acc': val_acc,
        'val_f1_macro': val_f1_macro,
        'time_sec': t_elapsed
    })

    print(f"  C={C} → val_acc={val_acc:.4f}, val_f1_macro={val_f1_macro:.4f} (耗时: {t_elapsed:.2f}s)")

    if val_f1_macro > best_lr_val_f1_macro:
        best_lr_val_f1_macro = val_f1_macro
        best_lr_val_acc = val_acc
        best_lr_C = C
        best_lr_model = lr

print(f"\n最佳 C: {best_lr_C}")
print(f"最佳验证集准确率: {best_lr_val_acc:.4f}")
print(f"最佳验证集 F1-macro: {best_lr_val_f1_macro:.4f}")

lr_best_result = evaluate_model(best_lr_model, X_train_vec, X_val_vec, y_train_part, y_val, "逻辑回归 (C={})".format(best_lr_C))

# ============ 算法四：多层感知机 (MLP) — 精简为 3 组 ============
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')

mlp_configs = [
    {'hidden_layer_sizes': (100,), 'alpha': 0.0001},
    {'hidden_layer_sizes': (100,), 'alpha': 0.001},
    {'hidden_layer_sizes': (100,), 'alpha': 0.01},
]

best_mlp_hidden = None
best_mlp_alpha = None
best_mlp_val_acc = 0
best_mlp_val_f1_macro = 0
best_mlp_model = None
mlp_tuning_results = []
mlp_loss_histories = {}

print("\n" + "=" * 60)
print("--- 算法四：多层感知机 (MLPClassifier) 超参数调优 ---")
print("=" * 60)
print("\n快速调参结果:")
for config in mlp_configs:
    hidden = config['hidden_layer_sizes']
    alpha = config['alpha']
    t_start = time.time()
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden,
        alpha=alpha,
        max_iter=200,
        solver='adam',
        random_state=RANDOM_STATE,
        verbose=False
    )
    mlp.fit(X_train_vec, y_train_part)
    t_end = time.time()
    t_elapsed = t_end - t_start
    
    y_val_pred = mlp.predict(X_val_vec)
    val_acc = accuracy_score(y_val, y_val_pred)
    val_f1_macro = f1_score(y_val, y_val_pred, average='macro')

    mlp_tuning_results.append({
        'hidden_layer_sizes': hidden,
        'alpha': alpha,
        'val_acc': val_acc,
        'val_f1_macro': val_f1_macro,
        'time_sec': t_elapsed
    })

    print(f"  hidden={hidden}, alpha={alpha} → val_acc={val_acc:.4f}, val_f1_macro={val_f1_macro:.4f} (耗时: {t_elapsed:.2f}s)")

    if val_f1_macro > best_mlp_val_f1_macro:
        best_mlp_val_f1_macro = val_f1_macro
        best_mlp_val_acc = val_acc
        best_mlp_hidden = hidden
        best_mlp_alpha = alpha
        best_mlp_model = mlp
        if hasattr(mlp, 'loss_curve_') and len(mlp.loss_curve_) > 0:
            mlp_loss_histories = {alpha: list(mlp.loss_curve_)}

print(f"\n最佳参数: hidden_layer_sizes={best_mlp_hidden}, alpha={best_mlp_alpha}")
print(f"最佳验证集准确率: {best_mlp_val_acc:.4f}")
print(f"最佳验证集 F1-macro: {best_mlp_val_f1_macro:.4f}")

mlp_best_result = evaluate_model(best_mlp_model, X_train_vec, X_val_vec, y_train_part, y_val, "MLP (hidden={}, alpha={})".format(best_mlp_hidden, best_mlp_alpha))

# ============ MLP 损失曲线图（仅最佳参数，一条线，节点密集真实） ============
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("\n--- 生成 MLP 损失曲线图 ---")
fig, ax = plt.subplots(figsize=(8, 5))
if best_mlp_alpha in mlp_loss_histories:
    loss_curve = mlp_loss_histories[best_mlp_alpha]
    x_vals = list(range(1, len(loss_curve) + 1))
    ax.plot(x_vals, loss_curve, label='alpha={}'.format(best_mlp_alpha), linewidth=1.5)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Loss')
    ax.set_title('MLP Loss Curve (hidden=(100,), alpha={}, solver=adam, max_iter=200)'.format(best_mlp_alpha))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('mlp_loss_curve.png', dpi=150)
    plt.close()
    print("📊 MLP 损失曲线已保存为 mlp_loss_curve.png")
else:
    print("⚠️ 最佳模型无 loss_curve，跳过画图")

# ============ 四种算法最终对比 & 选择最佳模型 ============
print("\n" + "=" * 60)
print("--- 四种算法最佳结果横向对比 ---")
print("=" * 60)

all_results = [nb_best_result, svm_best_result, lr_best_result, mlp_best_result]
comparison_table = []
for r in all_results:
    comparison_table.append({
        '算法': r['model_name'],
        'train_acc': r['train_acc'],
        'train_f1_macro': r['train_f1_macro'],
        'val_acc': r['val_acc'],
        'val_f1_macro': r['val_f1_macro'],
        'val_f1_micro': r['val_f1_micro'],
        'val_f1_weighted': r['val_f1_weighted']
    })

comparison_df = pd.DataFrame(comparison_table)
print(comparison_df.to_string(index=False))

best_overall = max(all_results, key=lambda x: x['val_f1_macro'])
print(f"\n🏆 最终最佳算法: {best_overall['model_name']}")
print(f"   训练集准确率: {best_overall['train_acc']:.4f}")
print(f"   验证集准确率: {best_overall['val_acc']:.4f}")
print(f"   验证集 F1-macro: {best_overall['val_f1_macro']:.4f}")

# ============ 新增：混淆矩阵（文本版 + 热力图） ============
from sklearn.metrics import confusion_matrix

print("\n" + "=" * 60)
print("--- 最佳模型混淆矩阵（验证集） ---")
print("=" * 60)

cm = confusion_matrix(y_val, best_overall['y_val_pred'])
cm_df = pd.DataFrame(cm, index=[f"真实{i}" for i in range(10)], columns=[f"预测{i}" for i in range(10)])
print(cm_df.to_string())

plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.title(f'Confusion Matrix - {best_overall["model_name"]}')
plt.colorbar()
tick_marks = np.arange(10)
plt.xticks(tick_marks, tick_marks)
plt.yticks(tick_marks, tick_marks)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
thresh = cm.max() / 2
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=9)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.close()
print("📊 混淆矩阵图已保存为 confusion_matrix.png")

# ============ 用最终最佳模型在【全量训练数据】上重新训练，预测测试集 ============
print("\n" + "=" * 60)
print("--- 使用最终最佳模型在全量训练集上重新训练并预测测试集 ---")
print("=" * 60)

vectorizer_full = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 1),
    stop_words='english'
)
X_train_full = vectorizer_full.fit_transform(X_train)
best_overall['model'].fit(X_train_full, y_train)

X_test_tfidf = vectorizer_full.transform(X_test_unlabeled)
final_predictions = best_overall['model'].predict(X_test_tfidf)

pd.DataFrame(final_predictions).to_csv('predictions.csv', index=False, header=False)
print("predictions.csv 已保存")
print("🏁 实验完成！")