from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import numpy as np

train_data = []
test_data = []
label = []
feature = []
c = 0

# 訓練データ読み込み
with open("train.dat") as f:
    for line in f:
        line = line.replace("[", "")
        line = line.replace("]", "")
        line = line.replace("\n", "")
        line = line.replace(",", "")
        line = line.split(" ")
        train_data.append(line)

# 訓練データをクラスと素性ベクトルに分離
for i in range(len(train_data)):
    x = train_data[i].pop(0)
    label.append(int(x))
feature = train_data

# テストデータ読み込み
with open("test.dat") as f:
    for line in f:
        line = line.replace("[", "")
        line = line.replace("]", "")
        line = line.replace("\n", "")
        line = line.replace(",", "")
        line = line.split(" ")
        test_data.append(line)

for i in range(len(feature)):
    for j in range(len(feature[i])):
        feature[i][j] = int(feature[i][j])

sc = StandardScaler()
# 訓練データの平均と標準偏差を計算
sc.fit(feature)
# 平均と標準偏差を用いて標準化
train_std = sc.transform(feature)
test_std = sc.transform(test_data)


# エポック数 40 、学習率 0.1 でパーセプトロンのインスタンスを生成
ppn = Perceptron(eta0=0.01, random_state=1)
# 訓練データをモデルに適合させる
ppn.fit(train_std, label)

# テストデータで予測を実施
pred = ppn.predict(train_std)
test_pred = ppn.predict(test_std)

TP = 0  # true positive
TN = 0  # true negative
FN = 0  # false negative
FP = 0  # false positive

for i in range(len(pred)):
    if label[i] == 1 and pred[i] == 1:
        TP += 1
    elif label[i] == 0 and pred[i] == 1:
        FP += 1
    elif label[i] == 1 and pred[i] == 0:
        FN += 1
    elif label[i] == 0 and pred[i] == 0:
        TN += 1

# print(TP, FP, FN, TN)

recall = TP / (TP + FN)
precision = TP / (TP + FP)

F = (2 * recall * precision) / (recall + precision)
print("パーセプトロン")
print("F値", F)

print('Misclassified examples: %d' % (label != pred).sum())

# 分類の正解率を表示
print('Accuracy: %.3f' % accuracy_score(label, pred))

out = test_pred.tolist()
test = []
with open("test.dat") as f:
    for line in f:
        test.append(line)

for i in range(len(out)):
    out[i] = str(out[i]) + " " + test[i]

print(out)
with open('test_pred.dat', 'w') as f:
    for line in out:
        f.write(line)

# print(label)
print(pred)
"""

class LogisticRegressionGD(object):
    def __init__(self, eta=0.05, n_iter=100, random_state=1):
        # 学習率の初期化、訓練回数の初期化、乱数シードを固定にする random_state
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []
        # 訓練回数分まで訓練データを反復処理
        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            # 誤差平方和のコストではなくロジスティック回帰のコストを計算することに注意
            cost = -y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output)))
            # エポックごとのコストを格納
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        # 総入力を計算 
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, z):
        # ロジスティックシグモイド活性化関数を計算 
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        # 1 ステップ後のクラスラベルを返す 
        return np.where(self.net_input(X) >= 0.0, 1, 0)
        # 以下に等しい :
        # return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)

X_train_01_subset = np.array(feature)
y_train_01_subset = np.array(label)

print(X_train_01_subset)
print(y_train_01_subset)

# ロジスティック回帰のインスタンスを生成
lrgd = LogisticRegressionGD(eta=0.05, n_iter=1000, random_state=1)
# モデルを訓練データに適合させる
lrgd.fit(X_train_01_subset, y_train_01_subset)

pred = lrgd.predict(y_train_01_subset)

for i in range(len(pred)):
    if label[i] == 1 and pred[i] == 1:
        TP += 1
    elif label[i] == 0 and pred[i] == 1:
        FP += 1
    elif label[i] == 1 and pred[i] == 0:
        FN += 1
    elif label[i] == 0 and pred[i] == 0:
        TN += 1

# print(TP, FP, FN, TN)

recall = TP / (TP + FN)
precision = TP / (TP + FP)

F = (2 * recall * precision) / (recall + precision)
print("ロジスティック回帰モデル")
print("F値", F)

print('Misclassified examples: %d' % (label != pred).sum())

# 分類の正解率を表示
print('Accuracy: %.3f' % accuracy_score(label, pred))

"""
