# Install packages in your environment before running:
# !pip install aif360 numpy pandas matplotlib seaborn scikit-learn

from aif360.datasets import CompasDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load COMPAS dataset
dataset_orig = CompasDataset()

# Split by protected attribute (race)
privileged_groups = [{'race': 1}]  # Caucasian
unprivileged_groups = [{'race': 0}]  # African-American

# Preprocessing: reweigh to mitigate bias
RW = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
dataset_transf = RW.fit_transform(dataset_orig)

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(dataset_transf.features)
y = dataset_transf.labels.ravel()

# Train logistic regression
clf = LogisticRegression(solver='liblinear')
clf.fit(X, y)
preds = clf.predict(X)

# Add predictions to dataset
dataset_transf_pred = dataset_transf.copy()
dataset_transf_pred.labels = preds

# Fairness metrics
metric = ClassificationMetric(
    dataset_transf, dataset_transf_pred,
    unprivileged_groups=unprivileged_groups,
    privileged_groups=privileged_groups
)

# Visualize False Positive Rate disparity
fpr_priv = metric.false_positive_rate(privileged=True)
fpr_unpriv = metric.false_positive_rate(privileged=False)

plt.bar(['Privileged (White)', 'Unprivileged (Black)'], [fpr_priv, fpr_unpriv], color=['green', 'red'])
plt.title('False Positive Rate by Race')
plt.ylabel('Rate')
plt.show()
