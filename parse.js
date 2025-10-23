const body = process.env.ISSUE_BODY || '';

// Отладочный вывод
console.error('=== DEBUG: Issue Body ===');
console.error(body);
console.error('=========================\n');

function extractField(body, label) {
  const regex = new RegExp(`### ${label}\\s*\\n\\s*(.+?)(?=\\n###|$)`, 's');
  const match = body.match(regex);
  const value = match ? match[1].trim() : '';
  console.error(`Field "${label}": "${value}"`);
  return value;
}

const fio = extractField(body, 'ФИО студента');
const group = extractField(body, 'Группа');
const slugRaw = extractField(body, 'Выберите тему');
const slug = slugRaw.includes('|') ? slugRaw.split('|')[0].trim() : slugRaw.trim();
const supervisor = extractField(body, 'Научный руководитель');
const access = extractField(body, 'Код доступа');

console.error(`\nExtracted slug: "${slug}"\n`);

console.log(`fio=${fio}`);
console.log(`group=${group}`);
console.log(`slug=${slug}`);
console.log(`supervisor=${supervisor}`);
console.log(`access_code=${access}`);
