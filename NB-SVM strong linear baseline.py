import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
# 读取数据
train = pd.read_csv('D:/kaggle/train.csv')
test = pd.read_csv('D:/kaggle/test.csv')
subm = pd.read_csv('D:/kaggle/sample_submission.csv')
train.head()
# 划分训练集和验证集
train_val, val = train_test_split(train, test_size=0.2, random_state=42)
# 特征提取
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
train['none'] = 1 - train[label_cols].max(axis=1)
train['comment_text'] = train['comment_text'].fillna("unknown")
# 特征工程
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s):
    return re_tok.sub(r' \1 ', s).split()

ngram_range = (1, 2)
vec = TfidfVectorizer(ngram_range=ngram_range, tokenizer=tokenize,
                      min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
                      smooth_idf=1, sublinear_tf=1)
# 训练和验证数据
train_term_doc = vec.fit_transform(train['comment_text'])
val_term_doc = vec.transform(val['comment_text'])
train_x = train_term_doc
val_x = val_term_doc
val_accuracies = []
# 定义模型
for j in label_cols:
    m = LogisticRegression()
    m.fit(train_x, train[j])  # 使用标签进行训练
    val_pred_proba = m.predict_proba(val_x)  # 获取概率
    train_pred_proba = m.predict_proba(train_x)  # 训练集概率
    val_accuracy = accuracy_score(val[j], val_pred_proba.argmax(axis=1))
    val_accuracies.append(val_accuracy)
    # 计算验证集准确率
    val_accuracy = accuracy_score(val[j], val_pred_proba.argmax(axis=1))  # 使用argmax获取预测标签
    print(f"Validation accuracy for {j}: {val_accuracy}")
# 绘制条形图
fig, ax = plt.subplots(figsize=(10, 6))
index = np.arange(len(label_cols))
bar_width = 0.35
opacity = 0.8

rects = ax.bar(index, val_accuracies, bar_width,
               alpha=opacity,
               color='b',
               label='Validation Accuracy')

ax.set_xlabel('Label')
ax.set_ylabel('Accuracy')
ax.set_title('Validation Accuracy for Each Label')
ax.set_xticks(index)
ax.set_xticklabels(label_cols)
ax.legend()

plt.tight_layout()
plt.show()
# 保存预测结果
submid = pd.DataFrame({'id': subm["id"]})
submission = pd.concat([submid, pd.DataFrame(preds, columns = label_cols)], axis=1)
file_path = 'D:/kaggle/submission_local.csv'
submission.to_csv(file_path, index=False)