from sklearn.linear_model import Perceptron as Percept

# 샘플과 레이블.
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [1, 1, 1, 0]        # xor
# y = [0, 1, 1, 1]        # or
# y = [0, 0, 0, 1]        # and

# 퍼셉트론 생성, tol= 종료 조건, random_state= 난수 시드
clf = Percept(tol=(1e-3), random_state=0, max_iter=3000)

# 학습
clf.fit(X, y)

print(clf.coef_, clf.intercept_)

# 테스트
print(clf.predict(X))
