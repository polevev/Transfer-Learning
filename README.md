# Neural Style Transfer Streamlit App

Данное веб‑приложение на Streamlit демонстрирует алгоритм нейронного переноса стиля (Neural Style Transfer). Пользователь загружает контент‑изображение и стиль‑изображение, выбирает количество итераций обучения, а модель генерирует новую картинку, сочетающую содержимое первого и художественный стиль второго.

# ⚙️ Возможности

- Перенос стиля с использованием предобученной VGG‑19

- Полоса прогресса и предпросмотр результата каждые 10 итераций

- Поддержка GPU (CUDA) при наличии

# 📋 Требования

- Python ≥ 3.8

- См. requirements.txt

- GPU (опционально)

- NVIDIA с CUDA 11+

Примечание : при запуске на CPU процесс может занять заметно больше времени.

# 🚀 Быстрый старт (локально)

### 1 — Клонируем репозиторий
$ git clone https://github.com/polevev/Transfer-Learning  
$ cd Transfer-Learning  

### 2 — Создаём и активируем виртуальное окружение
$ python -m venv venv  
$ source venv/bin/activate     # Windows: venv\Scripts\activate  

### 3 — Устанавливаем зависимости
(venv) $ pip install -r requirements.txt  

### 4 — Запускаем Streamlit‑приложение
(venv) $ streamlit run app.py  

После запуска откройте браузер и перейдите по адресу http://localhost:8501.

# 🐳 Запуск в Docker

Если предпочитаете контейнеризацию, используйте готовый Dockerfile или команды ниже.

Dockerfile (пример)

FROM python:3.10-slim  
WORKDIR /app  
COPY . .  
RUN pip install --no-cache-dir -r requirements.txt  
EXPOSE 8501  
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]  

# Сборка образа
$ docker build -t nst-app .

# Запуск контейнера
$ docker run -p 8501:8501 nst-app

Перейдите на http://localhost:8501 — приложение готово к работе.

# 📂 Структура проекта

.  
├── app.py            # Веб‑интерфейс Streamlit  
├── prjct.py          # Логика нейронного стиля (модель, обучение)  
├── requirements.txt  # Питон‑зависимости  
├── Dockerfile        # Контейнеризация (опционально)  
└── README.md         # Этот файл  

# 📝 Использование приложения

- Нажмите «Загрузите контент‑изображение» и выберите файл (JPEG/PNG/JPG).

- Нажмите «Загрузите стиль‑изображение» и выберите файл (JPEG/PNG/JPG).

- Установите количество итераций (рекомендуется < 50 (<= 3 минуты на GPU (3060 8гб), на CPU ~ 40 минут (2 минуты - эпоха))).

- Нажмите кнопку «Перенести стиль» и ожидайте завершения.

# ⚗️ Параметры обучения

| Параметр              | Значение по умолчанию | Где задаётся |
|-----------------------|-----------------------|--------------|
| `num_steps`           | 200                   | слайдер UI   |
| `style_weight`        | 1e5                   | `prjct.py`   |
| `content_weight`      | 1                     | `prjct.py`   |
| `Total Variation Loss`  | 1e-6                  | `prjct.py`   |

Измените значения в `prjct.py`, если нужен другой баланс стиля/контента.

# Результаты

Japanese flower

![image](https://github.com/user-attachments/assets/d87484cc-1970-43fd-93ea-3c787defe362)
![image](https://github.com/user-attachments/assets/53166868-603d-48fe-aa33-1ebb286ae187)
![image](https://github.com/user-attachments/assets/e62f8973-bacf-4cc3-ab09-8cdc8dd69ec1)

Starry night

![image](https://github.com/user-attachments/assets/df6b64d7-cd17-40e2-91bb-36b7975b07fc)
![image](https://github.com/user-attachments/assets/2a66feaa-8923-468f-b7bd-d488232ba96c)
![image](https://github.com/user-attachments/assets/2b109af9-3be1-4dfb-aa82-997f85baf156)
  
![image](https://github.com/user-attachments/assets/bdbb2b9b-f923-41c7-9708-e5833483cb9c)
![image](https://github.com/user-attachments/assets/c300eacb-e5c4-4783-b0f7-dca06bc281a1)
![image](https://github.com/user-attachments/assets/2ced2425-86e1-4ec6-a76b-2042c1f19674)












