#!/usr/bin/env node

/**
 * Проверяет, что уровень из формы, уровень группы и уровень темы совпадают.
 * Вход: путь к файлу темы (аргумент), FORM_LEVEL и GROUP (переменные окружения).
 * Выход: строки key=value для $GITHUB_OUTPUT. При ошибке — error=<код>, код выхода 1.
 */

const fs = require('fs');
const { topicLevel, formLevel, groupLevel, normalizeGroup } = require('./levels');

const out = (k, v) => console.log(`${k}=${v}`);
const fail = (code, extra = {}) => {
  out('error', code);
  for (const [k, v] of Object.entries(extra)) out(k, v);
  process.exit(1);
};

const form = formLevel(process.env.FORM_LEVEL);
if (!form) fail('no_form_level');

const group = groupLevel(process.env.GROUP);
if (!group.valid) fail('bad_group');
if (!group.level) fail('group_level_unknown');
if (group.level.id !== form.id) fail('group_level_mismatch', { group_level: group.level.topicValue });

const topic = topicLevel(fs.readFileSync(process.argv[2], 'utf8'));
if (!topic) fail('topic_no_level');
if (topic.id !== form.id) fail('topic_level_mismatch', { topic_level: topic.topicValue });

out('level', form.topicValue);
out('group', normalizeGroup(process.env.GROUP));
