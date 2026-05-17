import { readdir, readFile, writeFile } from 'fs/promises';
import { join, relative, dirname, resolve } from 'path';

const DOCS_DIR = './src/content/docs';
const OUTPUT = './src/data/backlinks.json';

const LINK_RE = /\[.*?\]\(([^)]+)\)|\[\[([^\]]+)\]\]/g;

async function* walk(dir) {
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const full = join(dir, entry.name);
    if (entry.isDirectory()) yield* walk(full);
    else if (entry.name.endsWith('.md') || entry.name.endsWith('.mdx')) yield full;
  }
}

function fileToSlug(filepath) {
  return '/' + relative(DOCS_DIR, filepath)
    .replace(/\\/g, '/')
    .replace(/\/index\.(md|mdx)$/, '/')
    .replace(/\.(md|mdx)$/, '/');
}

function normalizeTarget(target, sourceFile) {
  if (target.startsWith('/')) {
    return target.replace(/\.(md|mdx)$/, '').replace(/\/index$/, '/').replace(/\/?$/, '/');
  }
  const sourceDir = dirname(relative(DOCS_DIR, sourceFile));
  const parts = join(sourceDir, target).split('/').filter(Boolean);
  const resolved = [];
  for (const p of parts) {
    if (p === '..') resolved.pop();
    else if (p !== '.') resolved.push(p);
  }
  const slug = resolved.join('/').replace(/\.(md|mdx)$/, '').replace(/\/index$/, '');
  return '/' + slug + '/';
}

const backlinks = {};

for await (const file of walk(DOCS_DIR)) {
  const content = await readFile(file, 'utf8');
  const sourceSlug = fileToSlug(file);

  const titleMatch = content.match(/^title:\s*['"]?(.+?)['"]?\s*$/m);
  const title = titleMatch?.[1] ?? sourceSlug;

  for (const match of content.matchAll(LINK_RE)) {
    const raw = match[1] || match[2];
    if (!raw || raw.startsWith('http') || raw.startsWith('#')) continue;

    const target = raw.split('#')[0];
    if (!target) continue;

    const targetSlug = normalizeTarget(target, file);

    if (!backlinks[targetSlug]) backlinks[targetSlug] = [];
    if (!backlinks[targetSlug].find(b => b.slug === sourceSlug)) {
      backlinks[targetSlug].push({ slug: sourceSlug, title });
    }
  }
}

await writeFile(OUTPUT, JSON.stringify(backlinks, null, 2));
console.log(`Backlinks built: ${Object.keys(backlinks).length} pages with incoming links`);
