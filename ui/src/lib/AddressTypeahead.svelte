<script lang="ts">
	import { Combobox } from 'bits-ui';
	import { geocode, type LocationType, type Match, type Mode } from '@motis-project/motis-client';
	import { MapPinHouse as House, MapPin as Place } from '@lucide/svelte';
	import { parseCoordinatesToLocation, type Location } from './Location';
	import { language } from './i18n/translation';
	import maplibregl from 'maplibre-gl';
	import { getModeStyle, type LegLike } from './modeStyle';

	let {
		items = $bindable([]),
		selected = $bindable(),
		placeholder,
		name,
		place,
		placeBias,
		type,
		transitModes,
		onChange = () => {}
	}: {
		items?: Array<Location>;
		selected: Location;
		placeholder?: string;
		name?: string;
		place?: maplibregl.LngLatLike;
		placeBias?: number;
		type?: undefined | LocationType;
		transitModes?: Mode[];
		onChange?: (location: Location) => void;
	} = $props();

	let inputValue = $state('');
	let match = $state('');

	const getDisplayArea = (match: Match | undefined) => {
		if (match) {
			const matchedArea = match.areas.find((a) => a.matched);
			const defaultArea = match.areas.find((a) => a.default);
			if (matchedArea?.name.match(/^[0-9]*$/)) {
				matchedArea.name += ' ' + defaultArea?.name;
			}

			/* eslint-disable-next-line svelte/prefer-svelte-reactivity */
			const areas = new Set<number>();

			match.areas.forEach((a, i) => {
				if (a.matched || a.unique || a.default || a.adminLevel == 2 || a.adminLevel == 4) {
					if (a.name != match.name) {
						areas.add(i);
					}
				}
			});

			const sorted = Array.from(areas);
			sorted.sort((a, b) => b - a);

			return sorted.map((a) => match.areas[a].name).join(', ');
		}
		return '';
	};

	const getLabel = (match: Match) => {
		const displayArea = getDisplayArea(match);
		return displayArea ? match.name + ', ' + displayArea : match.name;
	};

	const updateGuesses = async () => {
		const coord = parseCoordinatesToLocation(inputValue);
		if (coord) {
			selected = coord;
			items = [];
			onChange(selected);
			return;
		}

		const pos = place ? maplibregl.LngLat.convert(place) : undefined;
		const biasPlace = pos ? { place: `${pos.lat},${pos.lng}` } : {};
		const baseQuery = {
			...biasPlace,
			placeBias,
			text: inputValue,
			language: [language],
			mode: transitModes
		};

		// If a specific type is requested, use single query (backward compatibility)
		if (type) {
			const { data: matches, error } = await geocode({
				query: {
					...baseQuery,
					type
				}
			});
			if (error) {
				console.error('TYPEAHEAD ERROR: ', error);
				return;
			}
			items = matches!.map((match: Match): Location => {
				return {
					label: getLabel(match),
					match
				};
			});
			/* eslint-disable-next-line svelte/prefer-svelte-reactivity */
			const shown = new Set<string>();
			items = items.filter((x) => {
				const entry = x.match?.type + x.label!;
				if (shown.has(entry)) {
					return false;
				}
				shown.add(entry);
				return true;
			});
			return;
		}

		// Make 3 parallel requests for STOP, PLACE, and ADDRESS
		const [stopsResult, placesResult, addressesResult] = await Promise.all([
			geocode({
				query: {
					...baseQuery,
					type: 'STOP'
				}
			}),
			geocode({
				query: {
					...baseQuery,
					type: 'PLACE'
				}
			}),
			geocode({
				query: {
					...baseQuery,
					type: 'ADDRESS'
				}
			})
		]);

		// Handle errors - continue with successful requests
		const allErrors = [stopsResult.error, placesResult.error, addressesResult.error].filter(
			(e) => e !== undefined
		);
		if (allErrors.length > 0) {
			console.error('TYPEAHEAD ERROR: ', allErrors);
		}

		// Convert matches to Locations and limit each type: 5 STOPS, 3 PLACES, 2 ADDRESSES
		const stops = (stopsResult.data ?? [])
			.slice(0, 5)
			.map((match: Match): Location => ({
				label: getLabel(match),
				match
			};
		});
	};

	const deserialize = (s: string): Location => {
		const x = JSON.parse(s);
		return {
			match: x,
			label: getLabel(x)
		};
	};

	$effect(() => {
		if (selected) {
			match = JSON.stringify(selected.match);
			inputValue = selected.label!;
		}
	});

	let ref = $state<HTMLElement | null>(null);
	$effect(() => {
		if (ref && inputValue) {
			(ref as HTMLInputElement).value = inputValue;
		}
	});

	let timer: number;
	$effect(() => {
		if (inputValue) {
			clearTimeout(timer);
			timer = setTimeout(() => {
				updateGuesses();
			}, 150);
		}
	});
</script>

{#snippet modeCircle(mode: Mode)}
	{@const modeIcon = getModeStyle({ mode } as LegLike)[0]}
	{@const modeColor = getModeStyle({ mode } as LegLike)[1]}
	<div
		style="background-color: {modeColor}; fill: white;"
		class="rounded-full flex items-center justify-center p-1"
	>
		<svg class="relative size-4 rounded-full">
			<use xlink:href={`#${modeIcon}`}></use>
		</svg>
	</div>
{/snippet}

<Combobox.Root
	type="single"
	allowDeselect={false}
	value={match}
	onValueChange={(e: string) => {
		if (e) {
			selected = deserialize(e);
			inputValue = selected.label!;
			onChange(selected);
		}
	}}
>
	<Combobox.Input
		{placeholder}
		{name}
		bind:ref
		class="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm transition-colors file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
		autocomplete="off"
		oninput={(e: Event) => (inputValue = (e.currentTarget as HTMLInputElement).value)}
		aria-label={placeholder}
		data-combobox-input={inputValue}
	/>
	{#if items.length !== 0}
		<Combobox.Portal>
			<Combobox.Content
				align="start"
				class="absolute top-2 w-[var(--bits-combobox-anchor-width)] z-10 overflow-hidden rounded-md border bg-popover text-popover-foreground shadow-md outline-none"
			>
				{#each items as item (item.match)}
					<Combobox.Item
						class="flex w-full cursor-default select-none rounded-sm py-4 pl-4 pr-2 text-sm outline-none data-[disabled]:pointer-events-none data-[highlighted]:bg-accent data-[highlighted]:text-accent-foreground data-[disabled]:opacity-50"
						value={JSON.stringify(item.match)}
						label={item.label}
					>
						<div class="flex items-center grow">
							<div class="size-6">
								{#if item.match?.type == 'STOP'}
									{@render modeCircle(item.match.modes?.length ? item.match.modes![0] : 'BUS')}
								{:else if item.match?.type == 'ADDRESS'}
									<House class="size-5" />
								{:else if item.match?.type == 'PLACE'}
									{#if !item.match?.category || item.match?.category == 'none'}
										<Place class="size-5" />
									{:else}
										<img
											src={`icons/categories/${item.match?.category}.svg`}
											alt={item.match?.category}
											class="size-5"
										/>
									{/if}
								{/if}
							</div>
							<div class="flex flex-col ml-4">
								<span class="font-semibold text-nowrap text-ellipsis overflow-hidden">
									{item.match?.name}
								</span>
								<span class="text-muted-foreground text-nowrap text-ellipsis overflow-hidden">
									{getDisplayArea(item.match)}
								</span>
							</div>
						</div>
						{#if item.match?.type == 'STOP'}
							<div class="mt-1 ml-4 flex flex-row gap-1.5 items-center">
								{#each item.match.modes! as mode, i (i)}
									{@render modeCircle(mode)}
								{/each}
							</div>
						{/if}
					</Combobox.Item>
				{/each}
			</Combobox.Content>
		</Combobox.Portal>
	{/if}
</Combobox.Root>
