<script lang="ts">
	import { Combobox, type Selected } from 'bits-ui';
	import { cn } from './utils';
	import { geocode, type Match } from './openapi';
	import { browser } from '$app/environment';

	let {
		placeholder,
		inputClass,
		name,
		onSelectedChange
	}: {
		placeholder: string | undefined;
		inputClass: string | undefined;
		name: string | undefined;
		onSelectedChange: ((match: Selected<Match> | undefined) => void) | undefined;
	} = $props();

	let inputValue = $state('');
	let touchedInput = $state(false);

	const language = browser ? navigator.languages.find((l) => l.length == 2) : '';

	type Item = { label: string; value: Match; area: string };

	let items = $state.raw<Array<Item>>([]);
	const updateGuesses = async () => {
		items = (
			await geocode<true>({
				query: { text: inputValue, language }
			})
		).data.map((match) => {
			const matchedArea = match.areas.find((a) => a.matched);
			const defaultArea = match.areas.find((a) => a.default);
			if (matchedArea?.name.match(/^[0-9]*$/)) {
				matchedArea.name += ' ' + defaultArea?.name;
			}
			let area = (matchedArea ?? defaultArea)!.name;
			if (area == match.name) {
				area = match.areas[0]!.name;
			}
			return { label: match.name + ', ' + area, area, value: match };
		});
		const shown = new Set<string>();
		items = items.filter((x) => {
			if (shown.has(x.label)) {
				return false;
			}
			shown.add(x.label);
			return true;
		});
	};

	let timer: number;
	$effect(() => {
		if (inputValue && touchedInput) {
			clearTimeout(timer);
			timer = setTimeout(() => {
				updateGuesses();
			}, 250);
		}
	});
</script>

<Combobox.Root {items} bind:inputValue bind:touchedInput {onSelectedChange}>
	<div class={cn('relative', inputClass)}>
		<Combobox.Input
			class="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm transition-colors file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
			{placeholder}
			aria-label={placeholder}
		/>
	</div>
	{#if items.length !== 0}
		<Combobox.Content
			sideOffset={12}
			class="relative z-50 min-w-[8rem] overflow-hidden rounded-md border bg-popover text-popover-foreground shadow-md outline-none"
		>
			{#each items as item (item.value)}
				<Combobox.Item
					class="relative flex w-full cursor-default select-none items-center rounded-sm py-2 pl-4 pr-2 text-sm outline-none data-[disabled]:pointer-events-none data-[highlighted]:bg-accent data-[highlighted]:text-accent-foreground data-[disabled]:opacity-50"
					value={item.value}
					label={item.label}
				>
					<span class="font-semibold text-nowrap text-ellipsis overflow-hidden">
						{item.value.name}
					</span>
					<span class="ml-2 text-muted-foreground text-nowrap text-ellipsis overflow-hidden">
						{item.area}
					</span>
				</Combobox.Item>
			{/each}
		</Combobox.Content>
	{/if}
	<Combobox.HiddenInput {name} />
</Combobox.Root>
