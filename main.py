import numpy as np
import matplotlib.pyplot as plt


# Класс перцептрона
class SingleLayerPerceptron:
    def __init__(self, init_weights='small_random'):
        self.w = None  # Вектор весов
        self.b = 0.0  # Скалярное смещение b
        self.init_weights = init_weights  # Сохраняем выбранный метод инициализации для fit()
        self.v_w = None  # Вектор "скорости" для момента по весам
        self.v_b = 0.0  # Скалярная "скорость" для момента по смещению

    def sigmoid(self, z):  # Функция активации
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def forward(self, X):  # Вычисляет z = Xw + b и применяет сигмоиду
        return self.sigmoid(X @ self.w + self.b)

    def compute_loss(self, y_true, y_pred, loss_type='bce',
                     l2_lambda=0.0):  # Функция потерь
        eps = 1e-15  # Малое число для защиты от log(0)
        y_pred = np.clip(y_pred, eps, 1 - eps)  # Ограничиваем вероятности
        if loss_type == 'bce':  # Бинарная кросс-энтропия: L = -mean[y*log(ŷ) + (1-y)*log(1-ŷ)]
            loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        elif loss_type == 'hinge':  # Hinge loss: L = mean[max(0, 1 - y_signed * z)]
            # Hinge требует z, поэтому вычисляем его заново через сохранённые веса
            z = np.dot(X_dummy := np.ones((1, self.w.shape[0])), self.w).flatten()
            loss = np.mean(np.maximum(0, 1 - (2 * y_true - 1) * z))
        else:
            loss = 0.0
        return loss + 0.5 * l2_lambda * np.sum(self.w ** 2)  # Добавляем L2-штраф: (λ/2)||w||² для регуляризации

    def fit(self, X_train, y_train, X_val, y_val, epochs=100, lr=0.1,  # Метод обучения (п.3 методички)
            batch_size=32, loss_type='bce', l2_lambda=0.0, momentum=0.0):
        m, d = X_train.shape  # m - число примеров, d - число признаков
        y_train_s = y_train.copy()  # Копия меток для безопасного изменения внутри цикла

        # Инициализация параметров
        if self.init_weights == 'zeros':  # Нулевая инициализация (ломает симметрию градиентов плохо)
            self.w = np.zeros(d)
        elif self.init_weights == 'small_random':  # Малые случайные веса
            self.w = np.random.randn(d) * 0.01
        elif self.init_weights == 'large_random':  # Большие веса (приводит к насыщению сигмоиды)
            self.w = np.random.randn(d) * 10.0
        else:
            self.w = np.random.randn(d) * 0.01  # Fallback на малые случайные
        self.b = 0.0  # Смещение всегда начинается с 0
        self.v_w = np.zeros(d)  # Обнуляем вектор момента для весов
        self.v_b = 0.0  # Обнуляем момент для смещения

        train_losses, val_losses = [], []  # Списки для хранения значений loss по эпохам (для графиков)

        for epoch in range(epochs):  # Цикл по эпохам
            idx = np.random.permutation(m)  # Генерируем случайную перестановку индексов для перемешивания данных
            X_sh, y_sh = X_train[idx], y_train_s[idx]  # Применяем перестановку к данным и меткам

            for start in range(0, m, batch_size):  # Цикл по мини-батчам с шагом batch_size
                end = start + batch_size  # Конец текущего батча
                X_b, y_b = X_sh[start:end], y_sh[start:end]  # Вырезаем текущий мини-батч

                z_b = X_b @ self.w + self.b  # Вычисляем линейную комбинацию z = Xw + b
                y_hat_b = self.sigmoid(z_b)  # Применяем сигмоиду: получаем ŷ

                # Вычисление градиентов
                if loss_type == 'bce':  # Градиенты для BCE + Sigmoid
                    dw = (1.0 / len(y_b)) * (X_b.T @ (y_hat_b - y_b)) + l2_lambda * self.w  # ∂L/∂w + L2
                    db = (1.0 / len(y_b)) * np.sum(y_hat_b - y_b)  # ∂L/∂b
                elif loss_type == 'hinge':  # Градиенты для Hinge loss
                    y_signed = 2 * y_b - 1  # Переводим метки из {0,1} в {-1, +1} для Hinge
                    margin = y_signed * z_b  # Вычисляем отступ (margin)
                    mask = margin < 1  # где отступ < 1, там есть ошибка (градиент != 0)
                    dw = -(1.0 / len(y_b)) * (X_b.T @ (mask * y_signed)) + l2_lambda * self.w  # Субградиент ∂L/∂w
                    db = -(1.0 / len(y_b)) * np.sum(mask * y_signed)  # Субградиент ∂L/∂b
                else:
                    raise ValueError("Unknown loss_type")  # Ошибка при неверном типе

                # Обновление параметров с моментом
                self.v_w = momentum * self.v_w + dw  # Обновляем "скорость": v = β*v + ∇
                self.v_b = momentum * self.v_b + db  # Аналогично для смещения
                self.w -= lr * self.v_w  # Обновляем веса: w = w - η*v
                self.b -= lr * self.v_b  # Обновляем смещение

            # Логирование потери после полной эпохи
            train_pred = self.forward(X_train)  # Прямой проход по всему train для оценки loss
            val_pred = self.forward(X_val)  # Прямой проход по всему val
            # Вычисляем и сохраняем все loss для стабильности графика
            train_losses.append(
                -np.mean(y_train * np.log(train_pred + 1e-15) + (1 - y_train) * np.log(1 - train_pred + 1e-15)))
            val_losses.append(-np.mean(y_val * np.log(val_pred + 1e-15) + (1 - y_val) * np.log(1 - val_pred + 1e-15)))

        return train_losses, val_losses  # Возвращаем историю потерь для построения графиков сходимости

    def predict_proba(self, X):  # Возвращает вероятности принадлежности к классу 1
        return self.forward(X)  # Просто вызывает прямой проход

    def predict(self, X, threshold=0.5):  # Возвращает бинарные метки 0/1
        return (self.predict_proba(X) >= threshold).astype(int)  # Применяем порог 0.5 по методичке


# Генератор данных
def gen_gaussian(n=500, noise=0.0):  # Генератор линейно разделимых данных (два гауссовых облака)
    X0 = np.random.multivariate_normal([2, 2], [[1, 0.5], [0.5, 1]], n // 2)  # Класс 0 центр [2,2]
    X1 = np.random.multivariate_normal([-2, -2], [[1, -0.5], [-0.5, 1]], n // 2)  # Класс 1 центр [-2,-2]
    X, y = np.vstack([X0, X1]), np.hstack([np.zeros(n // 2), np.ones(n // 2)])  # Объединяем признаки и метки
    if noise > 0:  # Если задан шум, случайно инвертируем метки с вероятностью noise
        flip = np.random.choice(n, int(n * noise), replace=False)  # Выбираем индексы для flip
        y[flip] = 1 - y[flip]  # Меняем 0->1 и 1->0
    return X, y


def gen_xor(n=500, noise=0.0):  # Генератор нелинейно разделимых данных
    X, y = [], []  # Временные списки для точек и меток
    for _ in range(n // 4):  # Генерируем по n/4 точек в каждый из 4 квадрантов
        X.append([np.random.uniform(0, 1), np.random.uniform(0, 1)])
        y.append(0)  # Квадрант 1
        X.append([np.random.uniform(-1, 0), np.random.uniform(0, 1)])
        y.append(1)  # Квадрант 2
        X.append([np.random.uniform(-1, 0), np.random.uniform(-1, 0)])
        y.append(0)  # Квадрант 3
        X.append([np.random.uniform(0, 1), np.random.uniform(-1, 0)])
        y.append(1)  # Квадрант 4
    X, y = np.array(X), np.array(y)  # Преобразуем списки в numpy-массивы
    if noise > 0:  # Добавляем шум аналогично gen_gaussian
        flip = np.random.choice(n, int(n * noise), replace=False)
        y[flip] = 1 - y[flip]
    return X, y


def gen_circle(n=500, noise=0.0):  # Генератор данных в виде круга
    r = np.sqrt(np.random.uniform(0, 1,
                                  n))  # Радиус - равномерное распределение и коррекция через sqrt для равномерности в круге
    theta = np.random.uniform(0, 2 * np.pi, n)  # равномерное распределение от 0 до 2π
    X = np.c_[r * np.cos(theta), r * np.sin(theta)]  # Преобразуем полярные координаты в декартовы (x, y)
    y = (r < np.sqrt(0.5)).astype(int)  # Метка 1, если точка внутри радиуса круга, иначе 0
    if noise > 0:  # Добавляем шум
        flip = np.random.choice(n, int(n * noise), replace=False)
        y[flip] = 1 - y[flip]
    return X, y


# Метрики + визуализация
def calc_metrics(y_true, y_prob, thr=0.5):  # Вычисляет основные метрики классификации (п.6 теории)
    y_pred = (y_prob >= thr).astype(int)  # Применяем порог для получения бинарных предсказаний
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))  # Считаем TP и FP
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FN = np.sum((y_true == 1) & (y_pred == 0))  # Считаем TN и FN
    prec = TP / (TP + FP + 1e-15)
    rec = TP / (TP + FN + 1e-15)  # Precision и Recall
    return {  # Возвращаем словарь метрик
        'acc': (TP + TN) / len(y_true), 'prec': prec, 'rec': rec,  # Accuracy, Precision, Recall
        'f1': 2 * prec * rec / (prec + rec + 1e-15)  # F1-score
    }


def calc_roc_auc(y_true, y_scores):  # Вычисляет ROC-кривую и AUC
    idx = np.argsort(-y_scores)  # Сортируем индексы по убыванию предсказанных вероятностей
    y_sorted = y_true[idx]  # Применяем сортировку к истинным меткам
    TPR = np.cumsum(y_sorted) / np.sum(y_sorted)  # True Positive Rate
    FPR = np.cumsum(1 - y_sorted) / np.sum(1 - y_sorted)  # False Positive Rate
    TPR = np.r_[0, TPR]
    FPR = np.r_[0, FPR]  # Добавляем начальную точку (0,0) для кривой
    auc = np.sum(np.diff(FPR) * (TPR[1:] + TPR[:-1]) / 2)  # Площадь под кривой методом трапеций
    return FPR, TPR, auc  # Возвращаем координаты кривой и значение AUC


def plot_boundary(model, X, y, title="Decision Boundary"):  # Визуализация разделяющей границы
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5  # Границы сетки по X1 с отступом 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5  # Границы сетки по X2
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))  # Создаём сетку 100x100
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)  # Предсказываем класс для каждой точки сетки
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')  # Рисуем закрашенные области классов
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolor='k', s=40)  # Накладываем исходные точки
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_errors(model, X, y, title="Errors"):  # Визуализация ошибочно классифицированных точек
    proba = model.predict_proba(X)  # Получаем вероятности
    pred = (proba >= 0.5).astype(int)  # Получаем предсказанные метки
    err_mask = pred != y  # True там, где предсказание != истина
    plt.scatter(X[~err_mask, 0], X[~err_mask, 1], c=y[~err_mask], cmap='coolwarm', alpha=0.7,
                label='Correct')  # Верные точки
    plt.scatter(X[err_mask, 0], X[err_mask, 1], facecolors='none', edgecolors='black', s=100, linewidth=2,
                label='Errors')  # Ошибки
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


# Кросс - валидация
def k_fold_cv(X, y, k=5, lr=0.1, batch_size=32, epochs=50):  # Функция k-кратной кросс-валидации
    idx = np.arange(len(X))  # Создаём массив индексов от 0 до n-1
    np.random.shuffle(idx)  # Перемешиваем индексы для случайного разбиения
    folds = np.array_split(idx, k)  # Делим индексы на k равных фолдов
    accs = []  # Список для хранения accuracy по каждому фолду
    for i in range(k):  # Проходим по каждому фолду как по валидационному
        val_idx, tr_idx = folds[i], np.concatenate([folds[j] for j in range(k) if j != i])  # Индексы val и train
        X_tr, y_tr, X_val, y_val = X[tr_idx], y[tr_idx], X[val_idx], y[val_idx]  # Разбиваем данные
        mu, sig = X_tr.mean(0), X_tr.std(0)
        sig[sig == 0] = 1  # Параметры стандартизации только по train
        X_tr, X_val = (X_tr - mu) / sig, (X_val - mu) / sig  # Применяем стандартизацию
        m = SingleLayerPerceptron()  # Создаём новую модель для каждого фолда
        m.fit(X_tr, y_tr, X_val, y_val, epochs=epochs, lr=lr, batch_size=batch_size)  # Обучаем
        accs.append(calc_metrics(y_val, m.predict_proba(X_val))['acc'])  # Сохраняем accuracy на val
    return np.mean(accs), np.std(accs)  # Возвращаем среднее и стандартное отклонение accuracy


if __name__ == "__main__":
    np.random.seed(42)

    print("Скрипт запущен...")

    print("\nОбучение и визуализация...")
    X_base, y_base = gen_gaussian(500)  # 500 точек, линейно разделимые
    tr_idx = np.random.choice(500, 350, replace=False)  # 350 индексов для обучения (70%)
    te_idx = np.array([i for i in range(500) if i not in tr_idx])  # Оставшиеся 150 для теста (30%)
    X_tr, y_tr, X_te, y_te = X_base[tr_idx], y_base[tr_idx], X_base[te_idx], y_base[te_idx]  # Фактическое разбиение
    mu, sig = X_tr.mean(0), X_tr.std(0)
    sig[sig == 0] = 1  # Среднее и станд. отклонение, защита от деления на 0
    X_tr, X_te = (X_tr - mu) / sig, (X_te - mu) / sig  # Преобразование данных
    X_val, y_val = X_tr[:100], y_tr[:100]  # Выделяем мини-валидацию из train для логирования loss

    model_base = SingleLayerPerceptron()  # Создаём экземпляр перцептрона
    tr_loss, val_loss = model_base.fit(X_tr, y_tr, X_val, y_val, epochs=100, lr=0.1,
                                       batch_size=32)  # Обучение с параметрами по умолчанию
    plt.plot(tr_loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.title('Loss (Default)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()

    plot_boundary(model_base, X_tr, y_tr, "Decision Boundary (Default)")  # Визуализация границы решений
    print(f"Тренировочная точность: {calc_metrics(y_tr, model_base.predict_proba(X_tr))['acc']:.3f}")
    print(f"Тестовая точность: {calc_metrics(y_te, model_base.predict_proba(X_te))['acc']:.3f}")

    # Задание 1. Собственный генератор данных
    print("\nТестирование на разных данных...")
    datasets = {  # Словарь с названиями и генераторами данных
        'Gaussian Linear': gen_gaussian(400),  # Линейно разделимые
        'XOR Non-linear': gen_xor(400),  # Нелинейно разделимые (XOR)
        'Circle Non-linear': gen_circle(400)  # Нелинейно разделимые (Круг)
    }
    for name, (X, y) in datasets.items():  # Перебираем наборы данных
        mu, sig = X.mean(0), X.std(0)
        sig[sig == 0] = 1  # Стандартизация
        X_std = (X - mu) / sig
        m = SingleLayerPerceptron()  # Новая модель
        m.fit(X_std, y, X_std, y, epochs=80, lr=0.1)  # Обучение на всём наборе (без разделения для наглядности)
        acc = calc_metrics(y, m.predict_proba(X_std))['acc']  # Оценка точности
        print(
            f"{name:<20} | Точность: {acc:.3f} | {'Линейно разделимо' if acc > 0.95 else 'Не линейно разделимо (перцептрон ограничен)'}")
        plot_boundary(m, X_std, y, f"Boundary: {name}")  # Граница для каждого набора

    # Задание 2. Дополнительные функции потерь и регуляризация
    print("\nHinge Loss и L2-регуляризация...")
    m_hinge = SingleLayerPerceptron()  # Модель для Hinge loss
    _, _ = m_hinge.fit(X_tr, y_tr, X_val, y_val, epochs=100, lr=0.05, loss_type='hinge')  # Обучение с Hinge
    m_l2 = SingleLayerPerceptron()  # Модель для L2 регуляризации
    _, _ = m_l2.fit(X_tr, y_tr, X_val, y_val, epochs=100, lr=0.1, l2_lambda=0.5)  # Обучение с λ=0.5
    print(
        f"Hinge Acc: {calc_metrics(y_te, m_hinge.predict_proba(X_te))['acc']:.3f} | ||w||={np.linalg.norm(m_hinge.w):.3f}")  # Точность и норма весов
    print(
        f"BCE+L2 Acc:{calc_metrics(y_te, m_l2.predict_proba(X_te))['acc']:.3f} | ||w||={np.linalg.norm(m_l2.w):.3f} (уменьшились)")  # Демонстрация эффекта L2

    # Задание 3. Метрики качества и анализ ошибок
    print("\nМетрики качества и ROC-кривая...")
    probs = model_base.predict_proba(X_te)  # Вероятности на тестовой выборке
    metrics = calc_metrics(y_te, probs)  # Вычисление всех метрик
    print(f"Precision: {metrics['prec']:.3f} | Recall: {metrics['rec']:.3f} | F1: {metrics['f1']:.3f}")
    fpr, tpr, auc = calc_roc_auc(y_te, probs)  # Расчёт ROC и AUC
    plt.plot(fpr, tpr, label=f'ROC (AUC={auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')  # ROC кривая + линия случайного угадывания
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid()
    plt.show()
    plot_errors(model_base, X_te, y_te, "Misclassified Points on Test")

    # Задание 4. Исследование сходимости градиентного спуска
    print("\nГрадиентный спуск с моментом...")
    betas = [0.0, 0.5, 0.9, 0.99]  # Различные коэффициенты импульса β
    plt.figure()  # Создаём новое окно графика
    for b in betas:  # Перебираем β
        m = SingleLayerPerceptron()  # Новая модель
        tr, _ = m.fit(X_tr, y_tr, X_val, y_val, epochs=100, lr=0.05, momentum=b)  # Обучение с моментом
        plt.plot(tr, label=f'β={b}')  # Строим линию loss
    plt.title('Loss with Momentum')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()

    # Задание 5. Кросс-валидация и подбор гиперпараметров
    print("\n5-кратная кросс-валидация и подбор гиперпараметров...")
    lrs = [0.001, 0.01, 0.1]  # Сетка скоростей обучения
    batches = [16, 64]  # Сетка размеров батчей
    best_acc, best_cfg = 0, None  # Хранилище лучшей конфигурации
    print("Grid Search CV:")
    for lr in lrs:  # Внешний цикл по lr
        for bs in batches:  # Внутренний цикл по batch_size
            mean_acc, std_acc = k_fold_cv(X_base, y_base, lr=lr, batch_size=bs, epochs=40)  # Запуск CV
            print(f"  lr={lr}, batch={bs} -> Acc={mean_acc:.3f} +- {std_acc:.3f}")
            if mean_acc > best_acc:  # Если текущая конфигурация лучше, сохраняем её
                best_acc, best_cfg = mean_acc, (lr, bs)
    print(f"Best: lr={best_cfg[0]}, batch={best_cfg[1]} (CV Acc={best_acc:.3f})")

    m_final = SingleLayerPerceptron()  # Создаём финальную модель
    m_final.fit(X_tr, y_tr, X_val, y_val, epochs=100, lr=best_cfg[0],
                batch_size=best_cfg[1])  # Обучаем на лучших гиперпараметрах
    print(f"Точность финального теста: {calc_metrics(y_te, m_final.predict_proba(X_te))['acc']:.3f}")