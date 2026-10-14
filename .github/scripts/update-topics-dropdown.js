#!/usr/bin/env node

/**
 * Обновляет выпадающие списки тем в формах резервации.
 * Каждая свободная тема попадает ровно в одну форму — по строке «**Уровень:**».
 * Тема без уровня не попадает ни в одну форму, а скрипт завершается с ошибкой,
 * чтобы куратор увидел это в Actions.
 */

const fs = require('fs');
const path = require('path');
const { LEVELS, topicLevel } = require('./levels');

const ROOT = path.join(__dirname, '../..');
const TOPICS_DIR = path.join(ROOT, 'diploma-info/topics');
const TEMPLATES_DIR = path.join(ROOT, '.github/ISSUE_TEMPLATE');
const EMPTY_OPTION = '— свободных тем этого уровня сейчас нет —';

const byLevel = Object.fromEntries(Object.keys(LEVELS).map(id => [id, []]));
const withoutLevel = [];

const files = fs.readdirSync(TOPICS_DIR)
  .filter(f => f.endsWith('.md') && f !== 'README.md' && f !== 'TEMPLATE.md');

for (const file of files) {
  const content = fs.readFileSync(path.join(TOPICS_DIR, file), 'utf8');
  const slug = file.replace(/\.md$/, '');

  const statusMatch = content.match(/\*\*Статус:\*\*\s*(.+)/i);
  const status = statusMatch ? statusMatch[1].trim() : '';
  if (!status.toLowerCase().includes('свободно')) continue;

  const titleMatch = content.match(/^#\s+(.+)$/m);
  const title = titleMatch ? titleMatch[1].trim() : slug;

  const level = topicLevel(content);
  if (!level) {
    withoutLevel.push(slug);
    continue;
  }
  byLevel[level.id].push({ slug, display: `${slug} | ${title}` });
}

const optionsRegex = /(- type: dropdown\s+id: slug\s+attributes:\s+label: [^\n]+\s+description: [^\n]+\s+options:\s*\n)([\s\S]*?)(\s+validations:)/;

for (const level of Object.values(LEVELS)) {
  const topics = byLevel[level.id].sort((a, b) => a.slug.localeCompare(b.slug));
  const options = topics.length ? topics.map(t => t.display) : [EMPTY_OPTION];
  const optionsLines = options.map(o => `        - ${o}`).join('\n');

  const file = path.join(TEMPLATES_DIR, level.templateFile);
  const content = fs.readFileSync(file, 'utf8');
  if (!optionsRegex.test(content)) {
    console.error(`❌ В ${level.templateFile} не найден блок dropdown со slug`);
    process.exit(1);
  }
  // Пробелы перед validations: схлопываются в один перевод строки — иначе каждый запуск добавлял пустую строку
  const updated = content.replace(optionsRegex, (_, head, _old, tail) => `${head}${optionsLines}\n${tail.replace(/^\s*\n/, '')}`);
  fs.writeFileSync(file, updated, 'utf8');
  console.log(`✅ ${level.templateFile}: свободных тем ${topics.length}`);
}

if (withoutLevel.length) {
  console.error(`❌ Свободные темы без строки «**Уровень:** бакалавриат|магистратура» не попали в формы: ${withoutLevel.join(', ')}`);
  process.exit(1);
}
