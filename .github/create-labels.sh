#!/bin/bash
# Скрипт для создания необходимых лейблов в репозитории

# Требуется GitHub CLI (gh)
# Установка: brew install gh

echo "Создаём лейблы для Issues..."

# Лейблы для типов заявок
gh label create "reservation" \
  --description "Заявка на резервацию темы" \
  --color "0E8A16" \
  --force

gh label create "topic-proposal" \
  --description "Предложение новой темы" \
  --color "1D76DB" \
  --force

gh label create "pending-review" \
  --description "Ожидает рассмотрения куратором" \
  --color "FBCA04" \
  --force

# Лейблы для статусов
gh label create "approved" \
  --description "Одобрено" \
  --color "0E8A16" \
  --force

gh label create "rejected" \
  --description "Отклонено" \
  --color "D93F0B" \
  --force

gh label create "needs-clarification" \
  --description "Требуются уточнения" \
  --color "FEF2C0" \
  --force

echo "✅ Лейблы созданы!"
