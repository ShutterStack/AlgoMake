# AlgoMake

*A meticulously crafted Machine Learning library, built entirely from scratch with NumPy. Dive deep into the mathematical brilliance of algorithms without hidden abstractions.*

---

## 📚 Table of Contents
- [Overview](#overview)
- [Why AlgoMake?](#why-algomake)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Gaussian Mixture Models (GMM)](#gaussian-mixture-models-gmm)
  - [Principal Component Analysis (PCA)](#principal-component-analysis-pca)
- [Running Tests](#running-tests)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## 🧠 Overview
**AlgoMake** is more than just a machine learning library; it's a journey into the fundamental mathematics that power AI. Each algorithm is implemented from first principles using only NumPy, providing a transparent and educational experience. It's designed for students, researchers, and practitioners who want to understand how these algorithms work — not just that they work.

---

## 💡 Why AlgoMake?
Most machine learning projects leverage high-level libraries. While efficient, this often obscures the underlying mechanics. **AlgoMake** distinguishes itself by:

- ✅ **Pure Python & NumPy**: No scikit-learn, TensorFlow, or PyTorch for core logic.
- ✅ **Deep Understanding**: Expose the core equations and their implementation.
- ✅ **Educational Value**: Ideal for learning, debugging, and experimenting with ML fundamentals.
- ✅ **Advanced Algorithms**: Beyond the basics, venturing into complex and often-skipped implementations.

---

## 🚀 Features
AlgoMake boasts a growing collection of hand-crafted algorithms, including:

### 📦 Core Algorithms (`algomake/models/`)

#### ✅ Gaussian Mixture Models (GMM)
- Full Expectation-Maximization (E-step & M-step) from scratch
- Custom multivariate Gaussian PDF
- Numerical stability considerations

#### 🧩 Support Vector Machines (SVM) *(Planned)*
- Dual formulation with Lagrange multipliers
- Hinge loss
- Kernel trick (linear, RBF)
- Sequential Minimal Optimization (SMO) algorithm

#### 📈 Linear Models
- Linear Regression *(Fully Implemented)*
- Logistic Regression *(Planned)*

#### 👥 K-Nearest Neighbors *(Fully Implemented)*
#### 🌲 Decision Trees *(Planned)*
#### 🔁 Ensemble Methods *(Planned: Bagging, Boosting)*
#### 🔍 Clustering Algorithms *(Planned: K-Means)*

### 🧹 Preprocessing & Dimensionality Reduction (`algomake/preprocessing/`)

#### 📉 Principal Component Analysis (PCA)
- Manual eigenvector/eigenvalue computation
- Covariance matrix implementation
- Dimensionality reduction pipeline

#### ⚖️ Standardization/Normalization *(Planned)*

### ⚙️ Utility Components
- `BaseEstimator`: Consistent API with `fit`, `predict`, `get_params`, `set_params`
- Metrics: Custom implementations of classification and regression metrics

---

## 📦 Installation
To get AlgoMake up and running, follow these steps:

```bash
git clone https://github.com/yourusername/algomake.git
cd algomake
pip install -e .
```

To install development dependencies:

```bash
pip install -e .[dev]
```

---

## 🧪 Usage

### Gaussian Mixture Models (GMM)
```python
import numpy as np
from algomake.models.gmm import GaussianMixture

np.random.seed(0)
mean_1 = np.array([2.0, 2.0])
cov_1 = np.array([[0.5, 0.2], [0.2, 0.5]])
data_1 = np.random.multivariate_normal(mean_1, cov_1, 100)

mean_2 = np.array([8.0, 8.0])
cov_2 = np.array([[0.7, -0.3], [-0.3, 0.7]])
data_2 = np.random.multivariate_normal(mean_2, cov_2, 100)

X_train = np.vstack((data_1, data_2))
gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(X_train)

print("Weights:", gmm.weights_)
print("Means:\n", gmm.means_)

X_new = np.array([[2.1, 1.9], [8.3, 7.8], [5.0, 5.0]])
labels = gmm.predict(X_new)
print("Predicted Labels:", labels)
```

### Principal Component Analysis (PCA)
```python
import numpy as np
from algomake.preprocessing.dimensionality_reduction import PCA

X = np.array([
    [2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2],
    [3.1, 3.0], [2.3, 2.7], [2.0, 1.6], [1.0, 1.1],
    [1.5, 1.6], [1.1, 0.9]
])

pca = PCA(n_components=1)
pca.fit(X)
X_reduced = pca.transform(X)
print("Transformed shape:", X_reduced.shape)
```

---

## 🧪 Running Tests
AlgoMake uses `pytest`. To run the test suite:

```bash
pytest
```

---

## 🤝 Contributing
We welcome contributions!

1. Fork the repo
2. Clone it:
   ```bash
   git clone https://github.com/yourusername/algomake.git
   ```
3. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. Add your changes + tests
5. Run tests: `pytest`
6. Format code: `black .` and `isort .`
7. Commit and push
8. Open a Pull Request 🚀

---

## 📄 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 📬 Contact
- GitHub Issues: [github.com/ShutterStack/AlgoMake/issues](https://github.com/ShutterStack/AlgoMake/issues)
- Email: patilarya3133@gmail.com
