import {
  Parser,
  between,
  char,
  choice,
  many,
  many1,
  possibly,
  sequenceOf,
  whitespace,
} from "arcsecond";

import { blockComment, lineComment } from "@/fbs/parser/comments";

export const whitespaceOrComments = many1(
  choice([whitespace, lineComment, blockComment]),
);

export const optionalWhitespaceOrComments = possibly(whitespaceOrComments);

export const surroundedBy = <T>(parser: Parser<T>) => between(parser)(parser);

export const whitespaceSurrounded = <T>(parser: Parser<T>) =>
  surroundedBy(optionalWhitespaceOrComments)(parser);

export const sepByWithOptionalTrailingSep =
  <S>(sepParser: Parser<S>) =>
  <T>(valueParser: Parser<T>): Parser<T[]> =>
    possibly(
      sequenceOf([
        valueParser,
        many(sequenceOf([sepParser, valueParser]).map(([, x]) => x)),
        possibly(sepParser),
      ]).map(([head, rest]): T[] => [head, ...rest]),
    ).map((x) => x ?? []);

export const commaSeparated = sepByWithOptionalTrailingSep(
  whitespaceSurrounded(char(",")),
);

export const betweenWithOptionalWhitespace = <L, R>(
  leftParser: Parser<L>,
  rightParser: Parser<R>,
) =>
  between(sequenceOf([leftParser, optionalWhitespaceOrComments]))(
    sequenceOf([optionalWhitespaceOrComments, rightParser]),
  );

export const betweenBraces = betweenWithOptionalWhitespace(
  char("{"),
  char("}"),
);
export const betweenBrackets = betweenWithOptionalWhitespace(
  char("["),
  char("]"),
);
export const betweenParens = betweenWithOptionalWhitespace(
  char("("),
  char(")"),
);
