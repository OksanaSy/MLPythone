# Homework


## Інсталяція залежностей

```bash
pip install -r requirements.txt
```


## Завдання 1

1. Допишіть в файлі `kfold.py` функції `kfold_cross_validation` та `evaluate_accuracy` для того щоб порахувати точність роботи K nearest neighbors класифікатора.

2. Порахуйте для різних `k` в `KNN` точність на **тестовому** датасеті і запишіть в `README.md`, `k` беріть з таблички нижче

 k | Accuracy
---|----------
 3 |  0.82
 4 |  0.83
 5 |  0.83
 6 |  0.83
 7 |  0.81
 9 |  0.8
10 |  0.81
15 |  0.79
20 |  0.78
21 |  0.78
40 |  0.73
41 |  0.73


Які можна зробити висновки про вибір `k`?

4
Accuracy on Test Data: 0.83
Average Cross-Validation Accuracy: 0.865
Test Accuracy vs. Avg Cross-Val Accuracy: -0.03249999999999997
5
Accuracy on Test Data: 0.83
Average Cross-Validation Accuracy: 0.8550000000000001
Test Accuracy vs. Avg Cross-Val Accuracy: -0.02750000000000008
6
Accuracy on Test Data: 0.83
Average Cross-Validation Accuracy: 0.86
Test Accuracy vs. Avg Cross-Val Accuracy: -0.030000000000000027

6 is best as accuracy is high, and less chances for overfitting
3. Знайшовши найкращий `k` змініть `num_folds` (в `main()`) та подивіться чи в середньому точність на валідаційних датасетах схожа з точністю на тестовому датасеті.
 num_folds = 10
Training data: (1000, 784) (1000,)
Test data: (400, 784) (400,)
 KNN with k = 6
Predicted 100/100 samples
Accuracy (Cross-Validation): 0.84
Predicted 100/100 samples
Accuracy (Cross-Validation): 0.86
Predicted 100/100 samples
Accuracy (Cross-Validation): 0.89
Predicted 100/100 samples
Accuracy (Cross-Validation): 0.89
Predicted 100/100 samples
Accuracy (Cross-Validation): 0.88
Predicted 100/100 samples
Accuracy (Cross-Validation): 0.83
Predicted 100/100 samples
Accuracy (Cross-Validation): 0.86
Predicted 100/100 samples
Accuracy (Cross-Validation): 0.87
Predicted 100/100 samples
Accuracy (Cross-Validation): 0.85
Predicted 100/100 samples
Accuracy (Cross-Validation): 0.89
Predicted 400/400 samples
Accuracy on Test Data: 0.83
Average Cross-Validation Accuracy: 0.87
Test Accuracy vs. Avg Cross-Val Accuracy: -0.04


num_folds=8
Training data: (1000, 784) (1000,)
Test data: (400, 784) (400,)
 KNN with k = 6
Predicted 125/125 samples
Accuracy (Cross-Validation): 0.84
Predicted 125/125 samples
Accuracy (Cross-Validation): 0.86
Predicted 125/125 samples
Accuracy (Cross-Validation): 0.87
Predicted 125/125 samples
Accuracy (Cross-Validation): 0.87
Predicted 125/125 samples
Accuracy (Cross-Validation): 0.86
Predicted 125/125 samples
Accuracy (Cross-Validation): 0.86
Predicted 125/125 samples
Accuracy (Cross-Validation): 0.84
Predicted 125/125 samples
Accuracy (Cross-Validation): 0.89
Predicted 400/400 samples
Accuracy on Test Data: 0.83
Average Cross-Validation Accuracy: 0.86
Test Accuracy vs. Avg Cross-Val Accuracy: -0.03

num_folds=6
Training data: (1000, 784) (1000,)
Test data: (400, 784) (400,)
 KNN with k = 6
Predicted 166/166 samples
Accuracy (Cross-Validation): 0.84
Predicted 166/166 samples
Accuracy (Cross-Validation): 0.89
Predicted 166/166 samples
Accuracy (Cross-Validation): 0.89
Predicted 166/166 samples
Accuracy (Cross-Validation): 0.84
Predicted 166/166 samples
Accuracy (Cross-Validation): 0.85
Predicted 170/170 samples
Accuracy (Cross-Validation): 0.88
Predicted 400/400 samples
Accuracy on Test Data: 0.83
Average Cross-Validation Accuracy: 0.86
Test Accuracy vs. Avg Cross-Val Accuracy: -0.03


num_folds=4
Training data: (1000, 784) (1000,)
Test data: (400, 784) (400,)
 KNN with k = 6
Predicted 250/250 samples
Accuracy (Cross-Validation): 0.85
Predicted 250/250 samples
Accuracy (Cross-Validation): 0.87
Predicted 250/250 samples
Accuracy (Cross-Validation): 0.84
Predicted 250/250 samples
Accuracy (Cross-Validation): 0.85
Predicted 400/400 samples
Accuracy on Test Data: 0.83
Average Cross-Validation Accuracy: 0.85
Test Accuracy vs. Avg Cross-Val Accuracy: -0.02

num_folds=2
Training data: (1000, 784) (1000,)
Test data: (400, 784) (400,)
 KNN with k = 6
Predicted 500/500 samples
Accuracy (Cross-Validation): 0.81
Predicted 500/500 samples
Accuracy (Cross-Validation): 0.83
Predicted 400/400 samples
Accuracy on Test Data: 0.83
Average Cross-Validation Accuracy: 0.82
Test Accuracy vs. Avg Cross-Val Accuracy: 0.01