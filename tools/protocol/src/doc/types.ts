export interface DocType {
  fqtn: string;
  title: string | undefined;
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
  summary?: string;
  description?: string;
  input?: string;
  output?: DocResponse[];
}

export interface DocResponse {
  type: string;
  description: string;
}

export interface Documentation {
  types: Map<string, DocType>;
  paths: DocPath[];
  // TODO: tags
}
