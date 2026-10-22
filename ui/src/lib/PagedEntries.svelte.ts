export class PagedEntries<Page, Entry> {
	private resolved = $state<Array<Page | undefined>>([]);
	private generation = 0;
	private entries: (page: Page) => Entry[];

	constructor(pages: () => Array<Promise<Page>>, entries: (page: Page) => Entry[]) {
		this.entries = entries;
		$effect(() => {
			const current = [...pages()];
			const generation = ++this.generation;
			this.resolved = current.map(() => undefined);
			current.forEach((p, i) =>
				p.then(
					(r) => {
						if (generation === this.generation) {
							this.resolved[i] = r;
						}
					},
					() => { }
				)
			);
		});
	}

	previous(page: Page, pageIndex: number, i: number): Entry | undefined {
		if (i > 0) {
			return this.entries(page)[i - 1];
		}
		for (let j = pageIndex - 1; j >= 0; --j) {
			const prev = this.resolved[j];
			const entries = prev ? this.entries(prev) : [];
			if (entries.length) {
				return entries[entries.length - 1];
			}
		}
		return undefined;
	}
}
