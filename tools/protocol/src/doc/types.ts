export interface DocType {
  fqtn: string;
  description: string | undefined;
  examples: any[];
  tags: string[];
  fields?: Map<string, DocField>;
}

export interface DocField {
  name: string;
  description?: string;
  examples?: any[];
}

export interface DocPath {
  path: string;
  summary?: string | undefined;
  description?: string | undefined;
  tags: string[];
  input?: string | undefined;
  output?: DocResponse | undefined;
}

export interface DocResponse {
  type: string;
  description: string;
}

export interface DocTagInfo {
  name: string;
  description: string;
}

export interface Documentation {
  types: Map<string, DocType>;
  paths: DocPath[];
  tags: DocTagInfo[];
}
