import numpy as np

def softmax_classifier(W, input, label, lamda):
    """
      Softmax Classifier

      Inputs have dimension D, there are C classes, a minibatch have N examples.
      (In this homework, D = 784, C = 10)

      Inputs:
      - W: A numpy array of shape (D, C) containing weights.
      - input: A numpy array of shape (N, D) containing a minibatch of data.
      - label: A numpy array of shape (N, C) containing labels, label[i] is a
        one-hot vector, label[i][j]=1 means i-th example belong to j-th class.
      - lamda: regularization strength, which is a hyerparameter.

      Returns:
      - loss: a single float number represents the average loss over the minibatch.
      - gradient: shape (D, C), represents the gradient with respect to weights W.
      - prediction: shape (N,), prediction[i]=c means i-th example belong to c-th class.
    """

    ############################################################################
    # TODO: Put your code here
    N = input.shape[0]  # 批量大小
    C = W.shape[1]  # 类别数（10）

    # 1. 计算得分 (N, C)
    scores = input.dot(W)

    # 2. Softmax 转换为概率
    # 数值稳定性：减去每行的最大值
    scores_shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores_shifted)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # (N, C)

    # 3. 计算数据损失（交叉熵）
    # 正确类别的概率: probs[range(N), label_argmax]
    label_argmax = np.argmax(label, axis=1)  # 从 one-hot 转为类别索引 (N,)
    correct_probs = probs[range(N), label_argmax]
    data_loss = -np.sum(np.log(correct_probs)) / N

    # 4. 正则化损失
    reg_loss = 0.5 * lamda * np.sum(W * W)

    # 5. 总损失
    loss = data_loss + reg_loss

    # 6. 计算梯度
    # 先计算 dscore = probs - label (N, C)
    dscore = probs - label
    # 梯度: (1/N) * X^T * dscore + lamda * W
    gradient = input.T.dot(dscore) / N + lamda * W

    # 7. 预测
    prediction = np.argmax(probs, axis=1)
    ############################################################################

    return loss, gradient, prediction
