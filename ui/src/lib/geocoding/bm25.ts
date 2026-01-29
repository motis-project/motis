/**
 * BM25 Scoring Utilities (ported from FlowRide backend).
 *
 * This is a lightweight field-aware BM25 implementation used to rank geocoding results
 * based on textual relevance, with boosts for exact/prefix matches and numeric tokens.
 */

export interface AddressAbbreviations {
	readonly [key: string]: string;
}

export const DEFAULT_ADDRESS_ABBREVIATIONS: AddressAbbreviations = {
	bd: 'boulevard',
	bvd: 'boulevard',
	boul: 'boulevard',
	av: 'avenue',
	avd: 'avenue',
	imp: 'impasse',
	'imp.': 'impasse',
	pl: 'place',
	plc: 'place',
	plce: 'place',
	r: 'rue',
	st: 'saint',
	ste: 'sainte',
	zi: 'zone industrielle',
	ch: 'chemin',
	che: 'chemin',
	sent: 'sentier',
	rte: 'route',
	rt: 'route',
	all: 'allee',
	al: 'allee',
	sq: 'square',
	pkg: 'parking',
	pking: 'parking',
	fg: 'faubourg',
	crs: 'cours',
	cd: 'cours',
	vlg: 'village',
	pt: 'pont',
	espl: 'esplanade',
	qu: 'quai',
	qt: 'quartier',
	res: 'residence',
	zac: 'zone d amenagement concerte',
	za: 'zone artisanale',
	zp: 'zone pietonne',
} as const;

export interface TokenizedFields {
	readonly [key: string]: readonly string[];
}

export interface FieldWeights {
	readonly [key: string]: number;
}

export interface Bm25Config {
	readonly k1: number;
	readonly b: number;
	readonly exactMatchBoost: number;
	readonly prefixMatchBoost: number;
	readonly numericExactBoost: number;
	readonly numericPrefixBoost: number;
	readonly fieldWeights: FieldWeights;
	readonly fieldOrder: readonly string[];
}

interface ScoringDocument {
	readonly index: number;
	readonly fieldTokens: TokenizedFields;
	readonly termFrequencies: ReadonlyMap<string, number>;
	readonly length: number;
}

interface ScoredResult<TItem> {
	readonly document: ScoringDocument;
	readonly item: TItem;
	readonly score: number;
	readonly typePriority: number;
}

export const normalizeText = (text: string): string => {
	if (text === '') return '';
	return text
		.toLowerCase()
		.normalize('NFD')
		.replace(/[\u0300-\u036f]/g, '')
		.replace(/[^\w\s]/gi, ' ')
		.replace(/\s+/g, ' ')
		.trim();
};

export const expandAbbreviations = (
	tokens: readonly string[],
	abbreviations: AddressAbbreviations = DEFAULT_ADDRESS_ABBREVIATIONS
): string[] => {
	const expanded: string[] = [];
	for (const token of tokens) {
		const replacement = abbreviations[token];
		if (replacement !== undefined) {
			expanded.push(...replacement.split(' '));
		} else {
			expanded.push(token);
		}
	}
	return expanded;
};

export const normalizeAndTokenize = (
	text: string,
	abbreviations: AddressAbbreviations = DEFAULT_ADDRESS_ABBREVIATIONS
): string[] => {
	const normalized = normalizeText(text);
	if (normalized === '') return [];
	const rawTokens = normalized.split(' ').filter((token) => token !== '');
	const expandedTokens = expandAbbreviations(rawTokens, abbreviations);
	return expandedTokens.filter((token) => token !== '');
};

export const isNumericToken = (token: string): boolean => {
	return /\d/.test(token);
};

export class Bm25Scorer<TItem> {
	private readonly config: Bm25Config;
	private readonly abbreviations: AddressAbbreviations;
	private documents: ScoringDocument[] = [];
	private averageDocumentLength = 0;

	constructor(config: Bm25Config, abbreviations: AddressAbbreviations = DEFAULT_ADDRESS_ABBREVIATIONS) {
		this.config = config;
		this.abbreviations = abbreviations;
	}

	index(items: readonly TItem[], fieldExtractor: (item: TItem) => TokenizedFields): this {
		this.documents = items.map((item, index) => {
			const fieldTokens = fieldExtractor(item);
			return this.buildScoringDocument(index, fieldTokens);
		});
		this.averageDocumentLength = this.calculateAverageDocumentLength();
		return this;
	}

	search(
		queryText: string,
		items: readonly TItem[],
		typePriorityExtractor?: (item: TItem) => number,
		prioritizeByType: boolean = false
	): TItem[] {
		if (this.documents.length === 0 || items.length === 0) return [...items];
		if (this.documents.length !== items.length) {
			throw new Error('Items array length must match indexed documents length');
		}

		const queryTokens = normalizeAndTokenize(queryText, this.abbreviations);
		if (queryTokens.length === 0) return [...items];
		const uniqueQueryTokens = Array.from(new Set(queryTokens));
		const idfScores = this.computeIdfScores(uniqueQueryTokens);
		if (this.averageDocumentLength <= 0) return [...items];

		const scoredResults: ScoredResult<TItem>[] = this.documents.map((document, index) => {
			const score = this.calculateBm25Score(document, uniqueQueryTokens, idfScores);
			const typePriority = typePriorityExtractor?.(items[index]) ?? 0;
			return { document, item: items[index], score, typePriority };
		});

		scoredResults.sort((a, b) => {
			if (a.score !== b.score) return b.score - a.score;
			if (prioritizeByType) {
				const typeDelta = a.typePriority - b.typePriority;
				if (typeDelta !== 0) return typeDelta;
			}
			return a.document.index - b.document.index;
		});

		return scoredResults.map((entry) => entry.item);
	}

	private computeIdfScores(queryTokens: readonly string[]): Map<string, number> {
		const docCount = this.documents.length;
		const idfScores = new Map<string, number>();
		for (const token of queryTokens) {
			let docFrequency = 0;
			for (const document of this.documents) {
				if (document.termFrequencies.has(token)) docFrequency += 1;
			}
			const numerator = docCount - docFrequency + 0.5;
			const denominator = docFrequency + 0.5;
			const idf = Math.log(1 + numerator / denominator);
			idfScores.set(token, idf);
		}
		return idfScores;
	}

	private calculateAverageDocumentLength(): number {
		if (this.documents.length === 0) return 0;
		let totalLength = 0;
		for (const document of this.documents) totalLength += document.length;
		return totalLength / this.documents.length;
	}

	private calculateBm25Score(
		document: ScoringDocument,
		queryTokens: readonly string[],
		idfScores: ReadonlyMap<string, number>
	): number {
		if (this.averageDocumentLength <= 0) return 0;

		const documentLength = document.length;
		let score = 0;

		for (const token of queryTokens) {
			const termFrequency = document.termFrequencies.get(token) ?? 0;
			if (termFrequency > 0) {
				const idf = idfScores.get(token) ?? 0;
				const normalization =
					this.config.k1 *
					(1 - this.config.b + this.config.b * (documentLength / this.averageDocumentLength));
				const bm25 = (termFrequency * (this.config.k1 + 1)) / (termFrequency + normalization);
				score += idf * bm25;
			}
			score += this.calculateMatchBoost(token, document.fieldTokens);
		}

		return score;
	}

	private calculateMatchBoost(token: string, fieldTokens: TokenizedFields): number {
		const exactFieldWeight = this.getBestFieldMatchWeight(token, fieldTokens, 'exact');
		const prefixFieldWeight =
			exactFieldWeight > 0 ? 0 : this.getBestFieldMatchWeight(token, fieldTokens, 'prefix');

		if (exactFieldWeight <= 0 && prefixFieldWeight <= 0) return 0;

		const numeric = isNumericToken(token);
		let boost = 0;

		if (exactFieldWeight > 0) {
			boost += this.config.exactMatchBoost * exactFieldWeight;
			if (numeric) boost += this.config.numericExactBoost * exactFieldWeight;
		}

		if (prefixFieldWeight > 0) {
			boost += this.config.prefixMatchBoost * prefixFieldWeight;
			if (numeric) boost += this.config.numericPrefixBoost * prefixFieldWeight;
		}

		return boost;
	}

	private getBestFieldMatchWeight(
		token: string,
		fieldTokens: TokenizedFields,
		matchType: 'exact' | 'prefix'
	): number {
		let bestWeight = 0;
		for (const field of this.config.fieldOrder) {
			const tokens = fieldTokens[field] ?? [];
			const weight = this.config.fieldWeights[field] ?? 0;
			if (matchType === 'exact') {
				if (tokens.includes(token)) bestWeight = Math.max(bestWeight, weight);
			} else if (tokens.some((value) => value.startsWith(token) && value !== token)) {
				bestWeight = Math.max(bestWeight, weight);
			}
		}
		return bestWeight;
	}

	private buildScoringDocument(index: number, fieldTokens: TokenizedFields): ScoringDocument {
		const termFrequencies = new Map<string, number>();
		let weightedLength = 0;

		for (const field of this.config.fieldOrder) {
			const tokens = fieldTokens[field] ?? [];
			const weight = this.config.fieldWeights[field] ?? 0;
			for (const token of tokens) {
				if (token === '') continue;
				const current = termFrequencies.get(token) ?? 0;
				termFrequencies.set(token, current + weight);
				weightedLength += weight;
			}
		}

		return { index, fieldTokens, termFrequencies, length: weightedLength };
	}
}

