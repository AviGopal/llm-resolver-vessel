// Runs named top-level declarations of index.ts in-process, against stubs, for tests.
//
// index.ts boots the daemon at import time (port, discovery registration), so its handlers cannot
// be imported. The declarations a test names are read out of index.ts's syntax tree, transpiled,
// and evaluated together in one scope: they call each other for real, and every other identifier
// they use resolves to a stub the test supplies, else to a JS global. An identifier with neither
// throws a ReferenceError naming it, so a missing stub fails loudly instead of reading undefined.
// It is the shipped code under test, not a copy of it.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import ts from "typescript";

const INDEX_PATH = join(import.meta.dir, "index.ts");

/** The source text of each named top-level function or const in index.ts, as a const binding. */
function declarationTexts(names: readonly string[], optional: ReadonlySet<string>): { texts: string[]; found: string[] } {
  const sf = ts.createSourceFile(INDEX_PATH, readFileSync(INDEX_PATH, "utf8"), ts.ScriptTarget.Latest, true, ts.ScriptKind.TS);
  const out: string[] = [];
  const found: string[] = [];
  for (const name of names) {
    let text: string | undefined;
    for (const st of sf.statements) {
      if (ts.isFunctionDeclaration(st) && st.name?.text === name) text = `const ${name} = ${st.getText(sf)};`;
      else if (ts.isVariableStatement(st) && st.declarationList.declarations.some((d) => ts.isIdentifier(d.name) && d.name.text === name)) {
        text = st.getText(sf);
      }
      if (text) break;
    }
    if (!text) {
      if (optional.has(name)) continue;
      throw new Error(`index.ts has no top-level declaration '${name}'`);
    }
    out.push(text);
    found.push(name);
  }
  return { texts: out, found };
}

/** Evaluate the named declarations with `stubs` as their free identifiers; returns them by name.
 * A name in `optional` that this revision of index.ts does not declare is skipped (a stub of that
 * name, if given, stands in for it). */
export function loadIndexDeclarations(
  names: readonly string[], stubs: Record<string, unknown>, optional: readonly string[] = [],
): Record<string, any> {
  const { texts, found } = declarationTexts(names, new Set(optional));
  const source = texts.join("\n");
  const js = ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2022 } }).outputText;
  const scope = new Proxy(stubs, {
    has: (_t, k) => typeof k === "string",
    get: (t, k) => {
      if (typeof k !== "string") return undefined;
      if (k in t) return t[k];
      if (k in globalThis) return (globalThis as Record<string, unknown>)[k];
      throw new ReferenceError(`unstubbed identifier '${k}' in index.ts declarations`);
    },
  });
  const body = `with (__scope) {\n${js}\nreturn { ${found.join(", ")} };\n}`;
  return new Function("__scope", body)(scope) as Record<string, any>;
}
