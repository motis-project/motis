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

	const getAreaNameByAdminLevel = (areas: readonly Match['areas'][number][], level: number): string => {
		const area = areas.find((entry) => entry.adminLevel === level);
		return area?.name ?? '';
	};

	const getRegion = (m: Match | undefined): string => {
		if (m === undefined) {
			return '';
		}
		const reversedAreas = [...(m.areas ?? [])].reverse();
		const city = reversedAreas.find((entry) => entry.adminLevel <= 8)?.name ?? '';
		const country = getAreaNameByAdminLevel(reversedAreas, 2);
		if (city !== '' && country !== '') {
			return `${city}, ${country}`;
		}
		if (city !== '') {
			return city;
		}
		return country;
	};

	const getLabel = (m: Match): string => {
		const region = getRegion(m);
		if (region !== '') {
			return `${m.name}, ${region}`;
		}
		return m.name;
	};

	const uniqueLocations = (matches: readonly Match[]): Match[] => {
		const seen: string[] = [];
		const unique: Match[] = [];
		for (const item of matches) {
			const region = getRegion(item);
			const key = `${item.type}|${item.name}|${region}`;
			if (!seen.includes(key)) {
				seen.push(key);
				unique.push(item);
			}
		}
		return unique;
	};

	const updateGuesses = async () => {
		if (inputValue.trim() === '') {
			items = [];
			return;
		}

		const coord = parseCoordinatesToLocation(inputValue);
		if (coord) {
			selected = coord;
			items = [];
			onChange(selected);
			return;
		}

		/*
		 * Default: same idea as transitous.org SearchBox — only `text` (plus language/mode).
		 * Send `place` + `placeBias` only after explicit geolocation (parent sets placeBias).
		 */
		const baseQuery = {
			text: inputValue,
			language: [language],
			mode: transitModes
		};

		let query:
			| { text: string; language: string[]; mode: Mode[] | undefined; type: LocationType | undefined }
			| {
					text: string;
					language: string[];
					mode: Mode[] | undefined;
					place: string;
					placeBias: number;
					type: LocationType | undefined;
			  };

		if (place !== undefined && placeBias !== undefined) {
			const pos = maplibregl.LngLat.convert(place);
			query = {
				...baseQuery,
				place: `${pos.lat},${pos.lng}`,
				placeBias,
				type
			};
		} else {
			query = { ...baseQuery, type };
		}

		const { data: matches, error } = await geocode({ query });
		if (error) {
			console.error('TYPEAHEAD ERROR: ', error);
			return;
		}

		const deduplicatedMatches = uniqueLocations(matches ?? []);
		items = deduplicatedMatches.map((entry): Location => ({
			label: getLabel(entry),
			match: entry
		}));
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
		} else {
			items = [];
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
									{getRegion(item.match)}
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
