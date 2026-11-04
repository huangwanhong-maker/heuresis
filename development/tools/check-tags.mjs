// The purposes of heuresis's draws are distinct: every tag("…") name used in the library's headers and checks
// is hashed as search/draws.hpp hashes it (FNV-1a, 32 bits), and two different names with one hash would share
// streams. Run from heuresis's root:  node development/tools/check-tags.mjs

import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';

const root = new URL('../..', import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1');
const files = [];
const walk = (dir) => {
  for (const name of readdirSync(dir)) {
    if (name === '.git' || name === 'build') continue;
    const p = join(dir, name);
    if (statSync(p).isDirectory()) walk(p);
    else if (/\.(hpp|cpp)$/.test(name)) files.push(p);
  }
};
walk(decodeURIComponent(root));

const fnv1a = (s) => {
  let h = 2166136261;
  for (const byte of new TextEncoder().encode(s)) h = Math.imul(h ^ byte, 16777619) >>> 0;
  return h >>> 0;
};

const names = new Map();  // name -> files
for (const f of files) for (const m of readFileSync(f, 'utf8').matchAll(/\btag\("([^"]*)"\)/g)) {
  if (!names.has(m[1])) names.set(m[1], new Set());
  names.get(m[1]).add(f);
}
const byHash = new Map();
for (const name of names.keys()) {
  const h = fnv1a(name);
  if (!byHash.has(h)) byHash.set(h, []);
  byHash.get(h).push(name);
}
const collisions = [...byHash.values()].filter((v) => v.length > 1);
for (const c of collisions) console.log(`FAIL  tags ${c.join(', ')} share the hash ${fnv1a(c[0]).toString(16)}`);
console.log(`${names.size} distinct tag names in ${files.length} files; ${collisions.length} collisions.`);
process.exit(collisions.length ? 1 : 0);
