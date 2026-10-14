#!/usr/bin/env node

/**
 * Скрипт для генерации страницы с зарезервированными темами
 * Парсит все файлы тем и извлекает информацию о студентах
 */

const fs = require('fs');
const path = require('path');
const { topicLevel, groupLevel } = require('./levels');

const TOPICS_DIR = path.join(__dirname, '../../diploma-info/topics');
const OUTPUT_FILE = path.join(__dirname, '../../diploma-info/RESERVED_TOPICS.md');
const REPO_BASE = 'https://github.com/VikZarip/mirea-iptip-diploma-topics/blob/main';

/**
 * Парсит файл темы и извлекает информацию о студентах
 */
function parseTopicFile(filePath, slug) {
  const content = fs.readFileSync(filePath, 'utf8');
  const lines = content.split('\n');
  
  // Извлекаем название темы (первая строка с #)
  const titleMatch = content.match(/^#\s+(.+)$/m);
  const title = titleMatch ? titleMatch[1].trim() : slug;
  
  // Проверяем статус
  const statusMatch = content.match(/\*\*Статус:\*\*\s*(.+?)$/m);
  const status = statusMatch ? statusMatch[1].trim() : '';
  
  if (status !== 'Зарезервирована') {
    return null;
  }
  
  // Ищем раздел "Закреплённые студенты"
  const studentsSection = content.match(/##\s*Закреплённые студенты[\s\S]*$/i);
  if (!studentsSection) {
    return null;
  }
  
  const students = [];
  const studentLines = studentsSection[0].split('\n');
  
  for (const line of studentLines) {
    // Формат 1: - **ФИО** (Группа) — дата — научный руководитель
    // Формат 2: ФИО (Группа) — научный руководитель (без маркера списка)
    
    // Пропускаем служебные строки
    if (line.includes('_Этот раздел заполняется') || 
        line.includes('<!-- Формат записи') ||
        line.trim() === '' ||
        line.startsWith('##')) {
      continue;
    }
    
    // Формат с маркером списка: - **ФИО** (Группа) — ...
    let match = line.match(/^-?\s*\*\*(.+?)\*\*\s*\((.+?)\)/);
    if (match) {
      const fio = match[1].trim();
      const group = match[2].trim();
      students.push({ fio, group });
      continue;
    }
    
    // Формат без маркера: ФИО (Группа) — ...
    match = line.match(/^([А-ЯЁа-яё\s]+)\s+\((.+?)\)/);
    if (match) {
      const fio = match[1].trim();
      const group = match[2].trim();
      students.push({ fio, group });
    }
  }
  
  if (students.length === 0) {
    return null;
  }
  
  return {
    slug,
    title,
    level: topicLevel(content),
    students
  };
}

/**
 * Магистр ли студент: по уровню темы, а для темы без уровня — по букве группы (ЭФМО → М).
 */
function isMaster(topic, group) {
  const level = topic.level || groupLevel(group).level;
  return level?.id === 'master';
}

/**
 * Собирает все зарезервированные темы
 */
function collectReservedTopics() {
  const files = fs.readdirSync(TOPICS_DIR);
  const reservedTopics = [];
  
  for (const file of files) {
    if (!file.endsWith('.md') || file === 'TEMPLATE.md' || file === 'README.md') {
      continue;
    }
    
    const slug = file.replace('.md', '');
    const filePath = path.join(TOPICS_DIR, file);
    
    try {
      const topicData = parseTopicFile(filePath, slug);
      if (topicData) {
        reservedTopics.push(topicData);
      }
    } catch (error) {
      console.error(`Ошибка при парсинге ${file}:`, error.message);
    }
  }
  
  return reservedTopics;
}

/**
 * Генерирует таблицу с зарезервированными темами
 */
function generateMarkdownTable(topics) {
  // Собираем все записи студентов
  const studentRecords = [];
  
  for (const topic of topics) {
    for (const student of topic.students) {
      studentRecords.push({
        fio: student.fio,
        group: student.group,
        isMaster: isMaster(topic, student.group),
        topicTitle: topic.title,
        topicSlug: topic.slug
      });
    }
  }
  
  // Разделяем на магистров и бакалавров
  const masters = studentRecords.filter(r => r.isMaster);
  const bachelors = studentRecords.filter(r => !r.isMaster);
  
  // Сортируем по ФИО
  masters.sort((a, b) => a.fio.localeCompare(b.fio, 'ru'));
  bachelors.sort((a, b) => a.fio.localeCompare(b.fio, 'ru'));
  
  let markdown = `# Зарезервированные темы

> Эта страница генерируется автоматически. Последнее обновление: ${new Date().toLocaleString('ru-RU', { timeZone: 'Europe/Moscow' })}

## Статистика

- **Всего зарезервировано тем:** ${topics.length}
- **Всего студентов:** ${studentRecords.length}
  - Магистров: ${masters.length}
  - Бакалавров: ${bachelors.length}

---

## Магистры

| ФИО | Группа | Тема |
|-----|--------|------|
`;

  // Генерируем таблицу магистров
  for (const record of masters) {
    const topicLink = `[${record.topicTitle}](${REPO_BASE}/diploma-info/topics/${record.topicSlug}.md)`;
    markdown += `| ${record.fio} | ${record.group} | ${topicLink} |\n`;
  }
  
  markdown += `\n---\n\n## Бакалавры\n\n`;
  markdown += `| ФИО | Группа | Тема |\n`;
  markdown += `|-----|--------|------|\n`;
  
  // Генерируем таблицу бакалавров
  for (const record of bachelors) {
    const topicLink = `[${record.topicTitle}](${REPO_BASE}/diploma-info/topics/${record.topicSlug}.md)`;
    markdown += `| ${record.fio} | ${record.group} | ${topicLink} |\n`;
  }
  
  markdown += `\n---\n\n`;
  markdown += `## Темы по алфавиту\n\n`;
  
  // Сортируем темы по названию
  const sortedTopics = [...topics].sort((a, b) => 
    a.title.localeCompare(b.title, 'ru')
  );
  
  for (const topic of sortedTopics) {
    markdown += `### [${topic.title}](${REPO_BASE}/diploma-info/topics/${topic.slug}.md)\n\n`;
    
    for (const student of topic.students) {
      const groupDisplay = isMaster(topic, student.group)
        ? `${student.group} (магистр)`
        : student.group;
      
      markdown += `- **${student.fio}** (${groupDisplay})\n`;
    }
    
    markdown += `\n`;
  }
  
  return markdown;
}

/**
 * Основная функция
 */
function main() {
  console.log('🔍 Сканирование файлов тем...');
  const topics = collectReservedTopics();
  
  console.log(`✅ Найдено зарезервированных тем: ${topics.length}`);
  
  if (topics.length === 0) {
    console.log('ℹ️  Нет зарезервированных тем для генерации страницы');
    return;
  }
  
  console.log('📝 Генерация markdown...');
  const markdown = generateMarkdownTable(topics);
  
  console.log(`💾 Сохранение в ${OUTPUT_FILE}...`);
  fs.writeFileSync(OUTPUT_FILE, markdown, 'utf8');
  
  console.log('✨ Готово!');
  console.log(`\nСтатистика:`);
  console.log(`- Тем: ${topics.length}`);
  console.log(`- Студентов: ${topics.reduce((sum, t) => sum + t.students.length, 0)}`);
}

main();
