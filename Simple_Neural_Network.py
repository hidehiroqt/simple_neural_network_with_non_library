import math
import matplotlib.pyplot as plt
import numpy as np


# シグモイド関数
def sigmoid(a):
    return 1.0 / (1.0 + math.exp(-a))


# ニューロン
class Neuron:
    input_sum = 0.0
    output = 0.0

    def setInput(self, inp):
        self.input_sum += inp

    def getOutput(self):
        self.output = sigmoid(self.input_sum)
        return self.output

    def reset(self):
        self.input_sum = 0
        self.output = 0


# ニューラルネットワーク
class NeuralNetwork:
    # 入力層と中間層の重み
    # w_im = [[1つ目の入力値に対する重み],[2つ目の入力値に対する重み],[バイアスに対する重み]]
    # 中間層が2つあるのでそれぞれの重みが2つずつある
    w_im = np.random.normal(0, scale=0.1, size=(3, 3))
    # 中間層と出力層の重み
    w_mo = np.random.normal(0, scale=0.1, size=3)

    # 各層の宣言
    # 2つのニューロンと1つのバイアス、数値がそのまま入るのでニューロン層を使わない
    input_layer = [0.0, 0.0, 1.0]
    middle_layer = [Neuron(), Neuron(), 1.0]
    output_layer = Neuron()

    # 実行
    def commit(self, input_data):

        # 各層のリセット
        for i in range(2):
            self.input_layer[i] = input_data[i]
            self.middle_layer[i].reset()

        self.output_layer.reset()

        # 入力層→中間層
        for i in range(2):
            for j in range(3):
                self.middle_layer[i].setInput(self.input_layer[j] * self.w_im[j][i])

        # 中間層→出力層
        for i in range(3):
            if i < 2:
                self.output_layer.setInput(self.middle_layer[i].getOutput() * self.w_mo[i])
            else:
                self.output_layer.setInput(self.middle_layer[i] * self.w_mo[i])

        return self.output_layer.getOutput()

    def learn(self, input_data):

        # 出力値
        output_data = self.commit([input_data[0], input_data[1]])
        # 正解値
        correct_value = input_data[2]
        # 学習係数
        k = 0.3

        # 出力層→中間層
        # Δ_mo = (出力値-正解値) * 出力値の微分(=シグモイド関数の微分)
        delta_w_mo = (correct_value - output_data) * output_data * (1.0 - output_data)
        # 更新前の重み
        old_w_mo = list(self.w_mo)
        # 修正量=Δ_mo * 中間層の出力値 * 学習係数
        self.w_mo[0] += delta_w_mo * self.middle_layer[0].output * k
        self.w_mo[1] += delta_w_mo * self.middle_layer[1].output * k
        self.w_mo[2] += delta_w_mo * self.middle_layer[2] * k

        # 中間層→入力層
        # Δ_im = Δ_mo * 中間出力の重み(更新前の重み) * 中間層の微分
        delta_w_im = [
            delta_w_mo * old_w_mo[0] * self.middle_layer[0].output * (1.0 - self.middle_layer[0].output),
            delta_w_mo * old_w_mo[1] * self.middle_layer[1].output * (1.0 - self.middle_layer[1].output)
        ]
        # 修正量 = delta_w_im * 入力層の値 * 学習係数
        for i in range(3):
            for j in range(2):
                self.w_im[i][j] += delta_w_im[j] * self.input_layer[i] * k


# 基準点(データの範囲を0.0-1.0の範囲に収めるため)
# refer_point_0 = 34.5
# refer_point_1 = 137.5
refer_point = [34.5, 137.5]

# ファイルの読み込み
training_data = []
training_data_file = open("training_data", "r")
for line in training_data_file:
    line = line.rstrip().split(",")
    training_data.append([float(line[0]) - refer_point[0], float(line[1]) - refer_point[1], int(line[2])])
training_data_file.close()

# ニューラルネットワークのインスタンス
neural_network = NeuralNetwork()

# 学習
print("初期値")
print("w_im:{}".format(neural_network.w_im))
print("w_mo:{}".format(neural_network.w_mo))

for t in range(0, 1000):
    for data in training_data:
        neural_network.learn(data)
    if (t+1) % 1000 == 0:
        print("epoch {}".format(t+1))
        print("w_im:{}".format(neural_network.w_im))
        print("w_mo:{}".format(neural_network.w_mo))

print("最終値")
print("w_im:{}".format(neural_network.w_im))
print("w_mo:{}".format(neural_network.w_mo))

# 実行
data_to_commit = [[34.6, 138.0], [34.6, 138.18], [35.4, 138.0], [34.98, 138.1], [35.0, 138.25], [35.4, 137.6], [34.98, 137.52], [34.5, 138.5], [35.4, 138.1]]
for data in data_to_commit:
    data[0] -= refer_point[0]
    data[1] -= refer_point[1]

position_tokyo_learned = [[], []]
position_kanagawa_learned = [[], []]

for data in data_to_commit:
    if neural_network.commit(data) < 0.5:
        for i in range(2):
            position_tokyo_learned[i].append(data[i] + refer_point[i])
    else:
        for i in range(2):
            position_kanagawa_learned[i].append(data[i] + refer_point[i])

# 訓練用データの表示の準備
position_tokyo_learning = [[], []]
position_kanagawa_learning = [[], []]
for data in training_data:
    if data[2] < 0.5:
        for i in range(2):
            position_tokyo_learning[i].append(data[i] + refer_point[i])
    else:
        for i in range(2):
            position_kanagawa_learning[i].append(data[i] + refer_point[i])

# プロット
plt.scatter(position_tokyo_learning[1], position_tokyo_learning[0], c="red", label="Tokyo_learn", marker="+")
plt.scatter(position_kanagawa_learning[1], position_kanagawa_learning[0], c="blue", label="Kanagawa_learn", marker="+")
plt.scatter(position_tokyo_learned[1], position_tokyo_learned[0], c="red", label="Tokyo", marker="o")
plt.scatter(position_kanagawa_learned[1], position_kanagawa_learned[0], c="blue", label="Kanagawa", marker="o")

plt.legend()
plt.show()
