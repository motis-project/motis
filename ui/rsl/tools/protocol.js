"use strict";
const fs = require("fs");
const path = require("path");

////////////////////////////////////////////////////////////////////////////////

// Tokens
const TOKENS = {
  IntegerConstant: Symbol("IntegerConstant"),
  FloatConstant: Symbol("FloatConstant"),
  BoolConstant: Symbol("BoolConstant"),
  StringConstant: Symbol("StringConstant"),
  Ident: Symbol("Ident"),
  Keyword: Symbol("Keyword"),
  PrimitiveType: Symbol("PrimitiveType"),
  Semicolon: Symbol("Semicolon"),
  Colon: Symbol("Colon"),
  Dot: Symbol("Dot"),
  Comma: Symbol("Comma"),
  LBrace: Symbol("LBrace"),
  RBrace: Symbol("RBrace"),
  LBracket: Symbol("LBracket"),
  RBracket: Symbol("RBracket"),
  LParen: Symbol("LParen"),
  RParen: Symbol("RParen"),
  Comment: Symbol("Comment"),
  Equals: Symbol("Equals"),
};

const KEYWORDS = [
  "namespace",
  "table",
  "struct",
  "enum",
  "union",
  "root_type",
  "attribute",
  "include",
];

const INT_TYPES = [
  "byte",
  "ubyte",
  "short",
  "ushort",
  "int",
  "uint",
  "long",
  "ulong",
  "int8",
  "uint8",
  "int16",
  "uint16",
  "int32",
  "uint32",
  "int64",
  "uint64",
];

const FLOAT_TYPES = ["float", "float32", "float64", "double"];
const NUMERIC_TYPES = [...INT_TYPES, ...FLOAT_TYPES];
const PRIMITIVE_TYPES = ["bool", "string", ...NUMERIC_TYPES];

class Tokenizer {
  static reWhitespace = /\s+/y;
  static reIdent = /[a-zA-Z_][a-zA-Z0-9_]*/y;
  static reHexInt = /[-+]?0[xX][0-9a-fA-F]+/y;
  static reDecFloat = /[-+]?((\.\d+)|(\d+\.\d*)|(\d+))([eE][-+]?\d+)?/y;
  static reString = /"[^"]*"/y;
  static reComment = /(\/\/.*)|(\/\*([^*]|\*(?!\/))*\*\/)/y;

  constructor(content) {
    this.content = content;
    this.pos = 0;
    this.len = this.content.length;
  }

  next() {
    while (
      this.match(Tokenizer.reWhitespace) ||
      this.match(Tokenizer.reComment)
    ) {}
    if (this.eof()) {
      return null;
    } else if (this.matchChar(";")) {
      return [TOKENS.Semicolon];
    } else if (this.matchChar(":")) {
      return [TOKENS.Colon];
    } else if (this.matchChar(".")) {
      return [TOKENS.Dot];
    } else if (this.matchChar(",")) {
      return [TOKENS.Comma];
    } else if (this.matchChar("{")) {
      return [TOKENS.LBrace];
    } else if (this.matchChar("}")) {
      return [TOKENS.RBrace];
    } else if (this.matchChar("[")) {
      return [TOKENS.LBracket];
    } else if (this.matchChar("]")) {
      return [TOKENS.RBracket];
    } else if (this.matchChar("(")) {
      return [TOKENS.LParen];
    } else if (this.matchChar(")")) {
      return [TOKENS.RParen];
    } else if (this.matchChar("=")) {
      return [TOKENS.Equals];
    } else if (this.match(Tokenizer.reIdent)) {
      const ident = this.matchedString;
      if (KEYWORDS.includes(ident)) {
        return [TOKENS.Keyword, ident];
      } else if (PRIMITIVE_TYPES.includes(ident)) {
        return [TOKENS.PrimitiveType, ident];
      } else if (ident === "true") {
        return [TOKENS.BoolConstant, true];
      } else if (ident === "false") {
        return [TOKENS.BoolConstant, false];
      } else {
        return [TOKENS.Ident, ident];
      }
    } else if (this.match(Tokenizer.reString)) {
      return [TOKENS.StringConstant, this.matchedString.slice(1, -1)];
    } else if (this.match(Tokenizer.reDecFloat)) {
      const str = this.matchedString;
      if (str.includes(".") || str.includes("e") || str.includes("E")) {
        return [TOKENS.FloatConstant, Number.parseFloat(str)];
      } else {
        return [TOKENS.IntegerConstant, Number.parseInt(str)];
      }
    } else if (this.match(Tokenizer.reHexInt)) {
      return [TOKENS.IntegerConstant, Number.parseInt(this.matchedString)];
    } else if (this.match(Tokenizer.reComment)) {
      return [TOKENS.Comment, this.matchedString];
    } else {
      throw `unexpected input: ${this.content.slice(
        this.pos,
        this.pos + 10
      )}...`;
    }
  }

  match(re) {
    if (this.eof()) {
      return null;
    }
    re.lastIndex = this.pos;
    const result = re.exec(this.content);
    if (result) {
      this.pos += result[0].length;
      this.matchedString = result[0];
      this.matchResult = result;
    }
    return result;
  }

  matchChar(c) {
    const cur = this.content[this.pos];
    if (cur === c) {
      ++this.pos;
      return true;
    }
    return false;
  }

  eof() {
    return this.pos >= this.len;
  }
}

class SchemaFile {
  constructor(fileName, content) {
    this.fileName = fileName;
    this.tokenizer = new Tokenizer(content);
    this.peeked = null;
    this.currentNamespace = "";
    this.includes = [];
    this.types = [];
    this.rootType = null;
  }

  parse() {
    while (!this.tokenizer.eof()) {
      let tok = this.consume(TOKENS.Keyword, true);
      if (tok === null) {
        break;
      }
      switch (tok[1]) {
        case "include": {
          const [, fn] = this.consume(TOKENS.StringConstant);
          this.consume(TOKENS.Semicolon);
          this.includes.push(fn);
          break;
        }
        case "namespace": {
          const ns = this.parseNamespace();
          this.currentNamespace = ns;
          break;
        }
        case "table":
        case "struct": {
          const tableOrStruct = tok[1];
          const [, name] = this.consume(TOKENS.Ident);
          this.consume(TOKENS.LBrace);
          const fields = [];
          do {
            tok = this.consume([TOKENS.Ident, TOKENS.RBrace]);
            if (tok[0] === TOKENS.Ident) {
              this.consume(TOKENS.Colon);
              const fieldType = this.parseType();
              const field = {
                name: tok[1],
                type: fieldType,
                attributes: [],
              };
              fields.push(field);
              tok = this.consume([
                TOKENS.Semicolon,
                TOKENS.Equals,
                TOKENS.LParen,
              ]);
              if (tok[0] === TOKENS.Equals) {
                field.defaultValue = this.parseLiteral(fieldType);
                tok = this.consume([TOKENS.Semicolon, TOKENS.LParen]);
              }
              if (tok[0] === TOKENS.LParen) {
                field.attributes = this.parseAttributes();
                tok = this.consume(TOKENS.Semicolon);
              }
            }
          } while (tok[0] !== TOKENS.RBrace);
          this.addType({
            type: tableOrStruct,
            name,
            namespace: this.currentNamespace,
            fields,
          });
          break;
        }
        case "enum": {
          const [, name] = this.consume(TOKENS.Ident);
          this.consume(TOKENS.Colon);
          const baseType = this.parseType();
          if (
            baseType.type !== "primitive" ||
            !INT_TYPES.includes(baseType.name)
          ) {
            throw "invalid enum base type: ${baseType}";
          }
          this.consume(TOKENS.LBrace);
          const values = [];
          let nextValue = 0;
          do {
            tok = this.consume([TOKENS.Ident, TOKENS.RBrace]);
            if (tok[0] === TOKENS.Ident) {
              const option = {
                name: tok[1],
                value: nextValue++,
                explicitValue: false,
              };
              tok = this.consume([TOKENS.Comma, TOKENS.RBrace, TOKENS.Equals]);
              if (tok[0] === TOKENS.Equals) {
                const [, optionValue] = this.consume(TOKENS.IntegerConstant);
                option.value = optionValue;
                option.explicitValue = true;
                nextValue = optionValue + 1;
                tok = this.consume([TOKENS.Comma, TOKENS.RBrace]);
              }
              values.push(option);
            }
          } while (tok[0] !== TOKENS.RBrace);
          this.addType({
            type: "enum",
            name,
            namespace: this.currentNamespace,
            values,
          });
          break;
        }
        case "union": {
          const [, name] = this.consume(TOKENS.Ident);
          this.consume(TOKENS.LBrace);
          const values = [];
          do {
            tok = this.consume([TOKENS.Ident, TOKENS.RBrace]);
            if (tok[0] === TOKENS.Ident) {
              this.defer(tok);
              const type = this.parseType();
              tok = this.consume([TOKENS.Comma, TOKENS.RBrace, TOKENS.Equals]);
              if (tok[0] === TOKENS.Equals) {
                const [, tagValue] = this.consume(TOKENS.IntegerConstant);
                type.tagValue = tagValue;
                tok = this.consume([TOKENS.Comma, TOKENS.RBrace]);
              }
              values.push(type);
            }
          } while (tok[0] !== TOKENS.RBrace);
          this.addType({
            type: "union",
            name,
            namespace: this.currentNamespace,
            values,
          });
          break;
        }
        case "root_type":
          this.rootType = this.parseType();
          this.consume(TOKENS.Semicolon);
          break;
        default:
          throw `unsupported keyword: ${tok[1]}`;
      }
    }
    this.tokenizer = null;
  }

  parseType() {
    let tok = this.next();
    switch (tok[0]) {
      case TOKENS.PrimitiveType:
        return { type: "primitive", name: tok[1] };
      case TOKENS.Ident: {
        let name = tok[1];
        for (tok = this.next(); tok[0] === TOKENS.Dot; tok = this.next()) {
          const [, nextPart] = this.consume(TOKENS.Ident);
          name = `${name}.${nextPart}`;
        }
        this.defer(tok);
        return {
          type: "custom",
          name,
          currentNamespace: this.currentNamespace,
        };
      }
      case TOKENS.LBracket: {
        const contentType = this.parseType();
        this.consume(TOKENS.RBracket);
        return { type: "array", contentType };
      }
      default:
        throw `invalid token ${String(tok[0])}, expected type`;
    }
  }

  parseNamespace() {
    let ns = "";
    loop: for (
      let tok = this.consume(TOKENS.Ident);
      tok !== null;
      tok = this.next()
    ) {
      switch (tok[0]) {
        case TOKENS.Ident:
          ns += tok[1];
          break;
        case TOKENS.Dot:
          ns += ".";
          break;
        case TOKENS.Semicolon:
          break loop;
        default:
          throw `unexpected token ${String(tok[0])} in namespace declaration`;
      }
    }
    if (ns.startsWith(".") || ns.endsWith(".")) {
      throw `invalid namespace: ${ns}`;
    }
    return ns;
  }

  parseLiteral(expectedType = undefined) {
    const allowed = [
      TOKENS.IntegerConstant,
      TOKENS.FloatConstant,
      TOKENS.BoolConstant,
      TOKENS.StringConstant,
    ];
    if (expectedType && expectedType.type === "custom") {
      allowed.push(TOKENS.Ident);
    }
    const [t, value] = this.consume(allowed);
    switch (t) {
      case TOKENS.IntegerConstant:
        return { type: "primitive", name: "int", value };
      case TOKENS.FloatConstant:
        return { type: "primitive", name: "float", value };
      case TOKENS.BoolConstant:
        return { type: "primitive", name: "bool", value };
      case TOKENS.StringConstant:
        return { type: "primitive", name: "string", value };
      case TOKENS.Ident:
        return { value, ...expectedType };
    }
  }

  parseAttributes() {
    let tok;
    const attributes = [];
    do {
      tok = this.consume([TOKENS.Ident, TOKENS.RParen]);
      if (tok[0] === TOKENS.Ident) {
        const attr = { name: tok[1] };
        tok = this.consume([TOKENS.Comma, TOKENS.RParen, TOKENS.Colon]);
        if (tok[0] === TOKENS.Colon) {
          attr.value = this.parseLiteral();
        }
        attributes.push(attr);
      }
    } while (tok[0] !== TOKENS.RParen);
    return attributes;
  }

  next(orEof = false) {
    const tok = this.peeked || this.tokenizer.next();
    if (tok === null && !orEof) {
      throw "unexpected eof";
    }
    this.peeked = null;
    return tok;
  }

  defer(tok) {
    if (this.peeked !== null) {
      throw "lookahead error";
    }
    this.peeked = tok;
  }

  consume(expected, orEof = false) {
    const tok = this.next(orEof);
    if (tok === null) {
      return null;
    }
    if (Array.isArray(expected)) {
      if (!expected.includes(tok[0])) {
        throw `unexpected token ${String(tok[0])}, expected one of: ${expected
          .map((t) => String(t))
          .join(", ")}`;
      }
    } else if (tok[0] !== expected) {
      throw `unexpected token ${String(tok[0])}, expected ${String(expected)}`;
    }
    return tok;
  }

  addType(type) {
    this.types.push(type);
  }
}

////////////////////////////////////////////////////////////////////////////////

function namespaced(ns, tn) {
  if (ns === "") {
    return tn;
  } else {
    return `${ns}.${tn}`;
  }
}

function namespacedTypeName(type) {
  return namespaced(type.namespace, type.name);
}

function getTypeNameWithoutNamespace(tn) {
  const nameParts = tn.split(".");
  return nameParts[nameParts.length - 1];
}

function getNamespace(fqtn) {
  return fqtn.split(".").slice(0, -1).join(".");
}

class FileSet {
  constructor(typeFilter) {
    this.files = [];
    this.types = new Map();
    this.namespaces = new Map();
    this.typeFilter = typeFilter.map((f) => f.split("."));
  }

  includeType(fqtn) {
    if (this.typeFilter.length === 0) {
      return true;
    }
    const parts = fqtn.split(".");
    outer: for (const filter of this.typeFilter) {
      let i = 0;
      for (const fp of filter) {
        if (i < parts.length && (fp === "*" || fp === parts[i])) {
          i++;
        } else {
          continue outer;
        }
      }
      if (i === parts.length) {
        return true;
      }
    }
    return false;
  }

  add(schemaFile) {
    this.files.push(schemaFile);
    for (const type of schemaFile.types) {
      const fqtn = namespacedTypeName(type);
      type.fileName = schemaFile.fileName;
      type.fqtn = fqtn;
      if (!this.includeType(fqtn)) {
        continue;
      }
      const prev = this.types.get(fqtn);
      if (prev) {
        throw `duplicate type: ${fqtn}, declared in ${prev.fileName} and ${type.fileName}`;
      }
      this.types.set(fqtn, type);
      const ns = getNamespace(fqtn);
      if (!this.namespaces.has(ns)) {
        this.namespaces.set(ns, []);
      }
      this.namespaces.get(ns).push(type);
    }
  }

  resolveTypes() {
    for (const [fqtn, type] of this.types) {
      switch (type.type) {
        case "table":
        case "struct": {
          let allResolved = true;
          for (const field of type.fields) {
            if (!this.resolveType(field.type)) {
              console.log(
                `\x1b[31mWarning: Unresolved type in ${
                  type.type
                } ${fqtn}: ${JSON.stringify(field.type)}\x1b[0m`
              );
              allResolved = false;
            }
            if (field.defaultValue) {
              field.defaultValue.resolved = field.type.resolved;
              field.defaultValue.fqtn = field.type.fqtn;
            }
          }
          type.resolved = allResolved;
          break;
        }
        case "union": {
          let allResolved = true;
          for (const t of type.values) {
            if (!this.resolveType(t)) {
              allResolved = false;
            }
          }
          type.resolved = allResolved;
          break;
        }
        case "enum":
          type.resolved = true;
          break;
      }
    }
  }

  resolveType(type) {
    switch (type.type) {
      case "custom": {
        let resolvedType = this.types.get(type.name);
        if (!resolvedType) {
          const namespace = type.currentNamespace.split(".");
          while (namespace.length > 0 && !resolvedType) {
            const fqtn = [...namespace, type.name].join(".");
            resolvedType = this.types.get(fqtn);
            namespace.pop();
          }
        }
        type.fqtn = resolvedType?.fqtn;
        type.resolved = type.fqtn != undefined;
        break;
      }
      case "array":
        type.resolved = this.resolveType(type.contentType);
        break;
      case "primitive":
        type.resolved = true;
        break;
    }
    return type.resolved;
  }
}

////////////////////////////////////////////////////////////////////////////////

function processFile(dir, fn, pathPrefix) {
  const relName = pathPrefix === "" ? fn : `${pathPrefix}/${fn}`;
  const fbs = fs.readFileSync(path.resolve(dir, fn), { encoding: "utf8" });
  const sf = new SchemaFile(relName, fbs);
  sf.parse();
  fileSet.add(sf);
}

function processDir(dir, pathPrefix) {
  for (const de of fs.readdirSync(dir, { withFileTypes: true })) {
    if (de.isFile()) {
      processFile(dir, de.name, pathPrefix);
    } else {
      const abs = path.resolve(dir, de.name);
      const rel = path.relative(protocolDir, abs).split(path.sep).join("/");
      processDir(abs, rel);
    }
  }
}

function getUnionTagTypeName(typeName) {
  return `${typeName}Type`;
}

function getConflictingAliasName(fqtn) {
  return fqtn.replaceAll(".", "_");
}

function getTypeScriptType(t, currentNs) {
  switch (t.type) {
    case "primitive": {
      if (t.name === "string") {
        return "string";
      } else if (t.name === "bool") {
        return "boolean";
      } else if (NUMERIC_TYPES.includes(t.name)) {
        return "number";
      }
      break;
    }
    case "custom": {
      const withoutNs = getTypeNameWithoutNamespace(t.name);
      const inCurrentNs = namespaced(currentNs, withoutNs);
      if (inCurrentNs !== t.fqtn && fileSet.types.has(inCurrentNs)) {
        return getConflictingAliasName(t.fqtn);
      }
      return withoutNs;
    }
    case "array": {
      return getTypeScriptType(t.contentType, currentNs) + "[]";
    }
  }
}

function getNamespaceFile(ns, ext = "") {
  if (ns.length === 0) {
    ns = "root";
  }
  const nsParts = ns.split(".");
  nsParts[nsParts.length - 1] += ext;
  return [
    nsParts.join("/"), // full path
    nsParts.slice(0, -1).join("/"), // dir
    nsParts[nsParts.length - 1], // file
  ];
}

function collectIncludes(ns, types) {
  const imported = new Map();

  const checkType = (t) => {
    if (t.type === "custom" && t.resolved) {
      const otherNs = getNamespace(t.fqtn);
      if (otherNs === ns) {
        return;
      }
      if (!imported.has(otherNs)) {
        imported.set(otherNs, new Set());
      }
      const imp = imported.get(otherNs);
      let impName = getTypeNameWithoutNamespace(t.fqtn);
      if (fileSet.types.has(namespaced(ns, impName))) {
        impName += ` as ${getConflictingAliasName(t.fqtn)}`;
      }
      imp.add(impName);
      const resolvedType = fileSet.types.get(t.fqtn);
      if (resolvedType.type === "union") {
        imp.add(getUnionTagTypeName(impName));
      }
    } else if (t.type === "array") {
      checkType(t.contentType);
    }
  };

  for (const t of types) {
    switch (t.type) {
      case "struct":
      case "table":
        t.fields.forEach((f) => checkType(f.type));
        break;
      case "union":
        t.values.forEach((v) => checkType(v));
        break;
    }
  }
  return imported;
}

function getRelativeImportPath(currentNs, importedNs) {
  const current = currentNs.split(".");
  const imported = importedNs.split(".");
  current.pop();
  const importedFile = imported.pop() || "root";
  while (current.length > 0 && imported.length > 0) {
    if (current[0] === imported[0]) {
      current.shift();
      imported.shift();
    } else {
      break;
    }
  }
  let path = [...current.map(() => ".."), ...imported, importedFile];
  if (path[0] !== "..") {
    path.unshift(".");
  }
  return path.join("/");
}

////////////////////////////////////////////////////////////////////////////////

const baseDir = process.cwd();
const argv = process.argv.slice(2);
if (argv.length > 1) {
  console.log(
    `usage: ${process.argv.slice(0, 2).join(" ")} [protocol.config.json]`
  );
  process.exit(1);
}
const configFile = path.resolve(
  baseDir,
  argv.length === 1 ? argv[0] : "protocol.config.json"
);
if (!fs.existsSync(configFile)) {
  console.log(`config file not found: ${configFile}`);
  process.exit(2);
}
const config = JSON.parse(fs.readFileSync(configFile, { encoding: "utf8" }));
if (!config.fbsDir || !config.tsOutputDir) {
  console.log(
    "invalid config file: ${configFile}\n\n${JSON.stringify(config, null, 2)}"
  );
  process.exit(3);
}

const protocolDir = path.resolve(baseDir, config.fbsDir);
const outputDir = path.resolve(baseDir, config.tsOutputDir);
const fileHeader = config.header ? `${config.header}\n\n` : "";

const fileSet = new FileSet(config.typeFilter || []);

console.log(`fbs input directory: ${protocolDir}`);
console.log(`ts output directory: ${outputDir}`);

processDir(protocolDir, "");
fileSet.resolveTypes();

for (const [fqtn, type] of fileSet.types) {
  if (!type.resolved) {
    console.log(
      `\x1b[31mWarning: Type with unresolved dependencies: ${type.type} ${fqtn} (${type.fileName})\x1b[0m`
    );
  }
}

const typeNames = new Map();
for (const [fqtn, type] of fileSet.types) {
  const parts = fqtn.split(".");
  const name = parts[parts.length - 1];
  if (typeNames.has(name)) {
    const otherType = fileSet.types.get(typeNames.get(name));
    console.log(
      `\x1b[33mWarning: Duplicate type name: ${name}\x1b[0m\n  - ${otherType.fqtn} (${otherType.fileName})\n  - ${fqtn} (${type.fileName})\n`
    );
  }
  typeNames.set(name, fqtn);
}

for (const [ns, types] of fileSet.namespaces) {
  const [, dirName, fileName] = getNamespaceFile(ns, ".ts");
  const dir = path.resolve(outputDir, dirName);
  const fn = path.resolve(dir, fileName);
  fs.mkdirSync(dir, { recursive: true });
  const out = fs.createWriteStream(fn);
  out.write(fileHeader);

  const includes = collectIncludes(ns, types);
  if (includes.size !== 0) {
    for (const [otherNs, importedTypes] of includes) {
      const importPath = getRelativeImportPath(ns, otherNs);
      out.write(
        `import { ${[...importedTypes].join(", ")} } from "${importPath}";\n`
      );
    }
    out.write("\n");
  }

  const writeEnum = (name, values, valueFormatter) => {
    if (values.length === 0) {
      // union, because of unresolved types
      out.write(`// empty type ${name}\n\n`);
      return;
    }
    out.write(`export type ${name} =`);
    for (const v of values) {
      out.write(`\n  | ${valueFormatter(v)}`);
    }
    out.write(";\n\n");
  };

  for (const t of types) {
    out.write(`// ${t.fileName}\n`);
    switch (t.type) {
      case "struct":
      case "table": {
        if (t.fields.filter((f) => f.type.resolved).length === 0) {
          out.write(
            "// eslint-disable-next-line @typescript-eslint/no-empty-interface\n"
          );
        }
        out.write(`export interface ${t.name} {`);
        for (const f of t.fields) {
          const typeName = getTypeScriptType(f.type, ns);
          if (f.type.type === "custom" && f.type.resolved) {
            const resolvedType = fileSet.types.get(f.type.fqtn);
            if (resolvedType.type === "union") {
              out.write(
                `\n  ${f.name}_type: ${getUnionTagTypeName(typeName)};`
              );
            }
          }
          out.write(
            `\n  ${f.type.resolved ? "" : "// "}${f.name}: ${typeName};`
          );
          const comments = [];
          if (f.defaultValue !== undefined) {
            comments.push(`default: ${f.defaultValue.value}`);
          }
          if (f.attributes) {
            for (const attr of f.attributes) {
              let comment = attr.name;
              if (attr.value) {
                comment += `: ${attr.value.value}`;
              }
              comments.push(comment);
            }
          }
          if (comments.length > 0) {
            out.write(` // ${comments.join(", ")}`);
          }
        }
        out.write("\n}\n\n");
        break;
      }
      case "enum": {
        writeEnum(t.name, t.values, (v) => `"${v.name}"`);
        break;
      }
      case "union": {
        const resolvedValues = t.values.filter((v) => v.resolved);
        writeEnum(t.name, resolvedValues, (v) => `${getTypeScriptType(v, ns)}`);
        writeEnum(
          getUnionTagTypeName(t.name),
          resolvedValues,
          (v) => `"${getTypeNameWithoutNamespace(v.name)}"`
        );
        break;
      }
    }
  }

  out.end();
}
