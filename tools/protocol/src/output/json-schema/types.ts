export type JSONSchemaType =
  | "null"
  | "boolean"
  | "object"
  | "array"
  | "number"
  | "string"
  | "integer";

export type JSONLiteral = string | number | boolean | null;

// just a rough subset
export type JSONSchema = {
  $schema?: string;
  $id?: string;
  $comment?: string;
  $ref?: string;
  $defs?: { [name: string]: JSONSchema };

  type?: JSONSchemaType | JSONSchemaType[];

  properties?: { [name: string]: JSONSchema };
  required?: string[];

  items?: JSONSchema;

  enum?: JSONLiteral[];

  allOf?: JSONSchema[];
  anyOf?: JSONSchema[];
  oneOf?: JSONSchema[];
  not?: JSONSchema;

  if?: JSONSchema;
  then?: JSONSchema;
  else?: JSONSchema;

  const?: JSONLiteral;

  minimum?: number;
  maximum?: number;
};
