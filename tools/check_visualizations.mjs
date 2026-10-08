// Smoke test for the HTML visualizations. For every .html file in the repo:
// - every inline <script> must parse as JavaScript (a syntax error would
//   leave the page blank);
// - external scripts and stylesheets may only come from an allowlist of
//   well-known CDNs (polyfill.io, for example, was taken over in 2024 and
//   served malicious code);
// - local src/href references must point to files that exist.
//
// Usage: node tools/check_visualizations.mjs   (no dependencies; exits 1 on
// any problem)
import { readdirSync, readFileSync, existsSync } from "node:fs";
import { dirname, join, relative } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const skipDirs = new Set([".git", "node_modules", ".venv", "build", "plugins", ".agents"]);
const allowedHosts = new Set([
  "cdn.tailwindcss.com",
  "cdn.jsdelivr.net",
  "cdnjs.cloudflare.com",
  "unpkg.com",
  "fonts.googleapis.com",
  "fonts.gstatic.com",
]);

function htmlFiles(dir) {
  const files = [];
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    if (entry.isDirectory()) {
      if (!skipDirs.has(entry.name)) files.push(...htmlFiles(join(dir, entry.name)));
    } else if (entry.name.endsWith(".html")) {
      files.push(join(dir, entry.name));
    }
  }
  return files;
}

const problems = [];
const files = htmlFiles(root).sort();
for (const file of files) {
  const name = relative(root, file);
  const html = readFileSync(file, "utf8");

  // Inline scripts: <script> without a src attribute.
  const scriptRe = /<script(?![^>]*\bsrc=)([^>]*)>([\s\S]*?)<\/script>/gi;
  let match;
  let inline = 0;
  while ((match = scriptRe.exec(html)) !== null) {
    const attrs = match[1];
    const code = match[2];
    if (/type\s*=\s*["'](?!text\/javascript|module)[^"']+["']/i.test(attrs)) continue; // JSON, templates
    inline += 1;
    try {
      new Function(code); // Parses without running.
    } catch (err) {
      problems.push(`${name}: inline script ${inline} does not parse: ${err.message}`);
    }
  }

  // External and local references in src= and href=.
  const refRe = /\b(?:src|href)\s*=\s*["']([^"']+)["']/gi;
  while ((match = refRe.exec(html)) !== null) {
    const ref = match[1];
    if (ref.startsWith("#") || ref.startsWith("data:") || ref.startsWith("mailto:")) continue;
    if (/^https?:\/\//i.test(ref)) {
      const host = new URL(ref).hostname;
      if (!allowedHosts.has(host)) problems.push(`${name}: loads from non-allowlisted host ${host} (${ref})`);
    } else if (!ref.includes("${")) {
      const target = join(dirname(file), ref.split(/[?#]/)[0]);
      if (!existsSync(target)) problems.push(`${name}: local reference not found: ${ref}`);
    }
  }
  console.log(`checked ${name} (${inline} inline scripts)`);
}

if (files.length === 0) problems.push("no .html files found");
if (problems.length > 0) {
  console.error(`\n${problems.length} problem(s):`);
  for (const p of problems) console.error(`  ${p}`);
  process.exit(1);
}
console.log(`\nAll ${files.length} visualizations passed.`);
