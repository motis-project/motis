<script lang="ts">
	import { Combobox } from 'bits-ui';
	import { cn } from './utils';
	import { geocode, type Match } from './openapi';
	import { browser } from '$app/environment';
	import Bus from 'lucide-svelte/icons/bus-front';
	import House from 'lucide-svelte/icons/map-pin-house';
	import Place from 'lucide-svelte/icons/map-pin';
	import type { Location } from './Location';
	import { GEOCODER_PRECISION } from './Precision';

	let {
		items = $bindable([]),
		selected = $bindable(),
		placeholder,
		class: className,
		name,
		theme
	}: {
		items?: Array<Location>;
		selected: Location;
		placeholder?: string;
		class?: string;
		name?: string;
		theme?: 'light' | 'dark';
	} = $props();

	let inputValue = $state('');
	let touchedInput = $state(false);

	const language = browser ? navigator.languages.find((l) => l.length == 2) : '';

	const getDisplayArea = (match: Match | undefined) => {
		if (match) {
			const matchedArea = match.areas.find((a) => a.matched);
			const defaultArea = match.areas.find((a) => a.default);
			if (matchedArea?.name.match(/^[0-9]*$/)) {
				matchedArea.name += ' ' + defaultArea?.name;
			}
			let area = (matchedArea ?? defaultArea)?.name;
			if (area == match.name) {
				area = match.areas[0]!.name;
			}
			return area;
		}
		return '';
	};

	const getLabel = (match: Match) => {
		const displayArea = getDisplayArea(match);
		return displayArea ? match.name + ', ' + displayArea : match.name;
	};

	const updateGuesses = async () => {
		items = (
			await geocode<true>({
				query: { text: inputValue, language }
			})
		).data.map((match: Match): Location => {
			return {
				label: getLabel(match),
				value: { match, precision: GEOCODER_PRECISION }
			};
		});
		const shown = new Set<string>();
		items = items.filter((x) => {
			if (shown.has(x.label!)) {
				return false;
			}
			shown.add(x.label!);
			return true;
		});
	};

	let timer: number;
	$effect(() => {
		if (inputValue && touchedInput) {
			clearTimeout(timer);
			timer = setTimeout(() => {
				updateGuesses();
			}, 150);
		}
	});
</script>

<Combobox.Root {items} bind:selected bind:inputValue bind:touchedInput>
	<div class={cn('relative', className)}>
		<Combobox.Input
			class="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm transition-colors file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
			{placeholder}
			aria-label={placeholder}
		/>
	</div>
	{#if items.length !== 0}
		<Combobox.Content
			sideOffset={12}
			class={cn(
				'absolute z-10 min-w-[8rem] overflow-hidden rounded-md border bg-popover text-popover-foreground shadow-md outline-none',
				theme
			)}
		>
			{#each items as item (item.value)}
				<Combobox.Item
					class="relative flex w-full cursor-default select-none items-center rounded-sm py-4 pl-4 pr-2 text-sm outline-none data-[disabled]:pointer-events-none data-[highlighted]:bg-accent data-[highlighted]:text-accent-foreground data-[disabled]:opacity-50"
					value={item.value}
					label={item.label}
				>
					{#if item.value.match?.type == 'STOP'}
						<Bus />
					{:else if item.value.match?.type == 'ADDRESS'}
						<House />
					{:else if item.value.match?.type == 'PLACE'}
						<Place />
					{/if}
					<span class="ml-4 font-semibold text-nowrap text-ellipsis overflow-hidden">
						{item.value.match?.name}
					</span>
					<span class="ml-2 text-muted-foreground text-nowrap text-ellipsis overflow-hidden">
						{getDisplayArea(item.value.match)}
					</span>
				</Combobox.Item>
			{/each}
		</Combobox.Content>
	{/if}
	<Combobox.HiddenInput {name} />
</Combobox.Root>
