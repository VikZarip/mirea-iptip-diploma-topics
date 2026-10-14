/**
 * Уровни обучения: единое правило для формы резервации, бота и страницы резерваций.
 *
 * Уровень темы задаётся строкой `**Уровень:** бакалавриат|магистратура` в файле темы.
 * Уровень студента — третьей буквой шифра группы: ЭФБО-01-23 → Б, ЭФМО-01-25 → М.
 */

const LEVELS = {
  bachelor: {
    id: 'bachelor',
    topicValue: 'бакалавриат',
    formValue: 'Бакалавриат',
    groupLetter: 'Б',
    templateFile: 'reserve-bachelor.yml',
    templateName: 'Резервация темы — бакалавриат',
    label: 'reservation-bachelor',
  },
  master: {
    id: 'master',
    topicValue: 'магистратура',
    formValue: 'Магистратура',
    groupLetter: 'М',
    templateFile: 'reserve-master.yml',
    templateName: 'Резервация темы — магистратура',
    label: 'reservation-master',
  },
};

const GROUP_RE = /^[А-ЯЁ]{2}([А-ЯЁ])[А-ЯЁ]-\d{2}-\d{2}(\d{2})?$/;

function byValue(field, value) {
  const v = (value || '').trim().toLowerCase();
  return Object.values(LEVELS).find(l => l[field].toLowerCase() === v) || null;
}

/** Уровень темы по содержимому файла или null, если строки нет или значение неизвестно. */
function topicLevel(content) {
  const m = content.match(/^\*\*Уровень:\*\*\s*(.+?)\s*$/m);
  return m ? byValue('topicValue', m[1]) : null;
}

/** Уровень из поля «Уровень» формы резервации. */
function formLevel(value) {
  return byValue('formValue', value);
}

/** Нормализованная группа: верхний регистр, без пробелов, длинное тире → дефис. */
function normalizeGroup(group) {
  return (group || '').toUpperCase().replace(/\s+/g, '').replace(/[–—]/g, '-');
}

/**
 * Уровень студента по шифру группы.
 * Возвращает { level, valid }: valid=false — группа не похожа на шифр МИРЭА,
 * level=null при valid=true — буква уровня не Б и не М (например, специалитет).
 */
function groupLevel(group) {
  const m = normalizeGroup(group).match(GROUP_RE);
  if (!m) return { level: null, valid: false };
  return { level: byValue('groupLetter', m[1]), valid: true };
}

module.exports = { LEVELS, topicLevel, formLevel, groupLevel, normalizeGroup };
