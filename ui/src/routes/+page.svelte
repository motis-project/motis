<script lang="ts">
	import X from 'lucide-svelte/icons/x';
	import { getStyle } from '$lib/map/style';
	import Map from '$lib/map/Map.svelte';
	import Control from '$lib/map/Control.svelte';
	import SearchMask from './SearchMask.svelte';
	import { type Location } from '$lib/Location';
	import { Card } from '$lib/components/ui/card';
	import type { Selected } from 'bits-ui';
	import { type Itinerary, plan, type PlanResponse } from '$lib/openapi';
	import ItineraryList from './ItineraryList.svelte';
	import ConnectionDetail from './ConnectionDetail.svelte';
	import { Button } from '$lib/components/ui/button';
	import ItineraryGeoJson from './ItineraryGeoJSON.svelte';
	import maplibregl from 'maplibre-gl';
	import { browser } from '$app/environment';
	import { cn } from '$lib/utils';
	import ThemeToggle from '$lib/ThemeToggle.svelte';

	const updateBodyTheme = (theme: 'dark' | 'light') => {
		document.documentElement.classList.remove('dark');
		if (theme === 'dark') {
			document.documentElement.classList.add('dark');
		}
	};
	let theme = $state<'dark' | 'light'>(
		browser && window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches
			? 'dark'
			: 'light'
	);
	if (browser) {
		window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (event) => {
			theme = event.matches ? 'dark' : 'light';
		});
	}
	$effect(() => {
		updateBodyTheme(theme);
	});

	let level = $state(0);
	let zoom = $state(15);
	let bounds = $state<maplibregl.LngLatBoundsLike>();
	let map = $state<maplibregl.Map>();

	let from = $state<Location>({ label: '', value: {} });
	let to = $state<Location>({ label: '', value: {} });
	let dateTime = $state<Date>(new Date());
	let timeType = $state<string>('departure');
	let profile = $state<Selected<string>>({ value: 'foot', label: 'Foot' });

	const pad = (x: number) => ('0' + x).slice(-2);
	const toPlaceString = (l: Location) => {
		if (l.value.level) {
			return `${l.value.match?.lat},${l.value.match?.lon},${l.value.level}`;
		} else {
			return `${l.value.match?.lat},${l.value.match?.lon},0`;
		}
	};
	let baseQuery = $derived(
		from.value.match && to.value.match
			? {
					query: {
						date: `${pad(dateTime.getUTCMonth() + 1)}-${pad(dateTime.getUTCDate())}-${dateTime.getUTCFullYear()}`,
						time: `${pad(dateTime.getUTCHours())}:${pad(dateTime.getUTCMinutes())}`,
						fromPlace: toPlaceString(from),
						toPlace: toPlaceString(to),
						wheelchair: profile.value === 'wheelchair',
						arriveBy: timeType === 'arrival',
						timetableView: true
					}
				}
			: undefined
	);
	let routingResponses = $state<Array<Promise<PlanResponse>>>([]);
	$effect(() => {
		if (baseQuery) {
			routingResponses = [plan<true>(baseQuery).then((response) => response.data)];
			selectedItinerary = undefined;
		}
	});

	let selectedItinerary = $state<Itinerary>();
</script>

<Map
	bind:map
	bind:bounds
	bind:zoom
	transformRequest={(url: string) => {
		if (url.startsWith('/')) {
			return { url: `https://europe.motis-project.de/tiles${url}` };
		}
	}}
	center={[8.652235, 49.876908]}
	class={cn('h-screen', theme)}
	style={getStyle(theme, level)}
>
	<Control position="top-right">
		<ThemeToggle bind:theme />
	</Control>

	<Control position="top-left">
		<Card class="w-[500px] overflow-y-auto overflow-x-hidden bg-background rounded-lg">
			<SearchMask bind:from bind:to bind:dateTime bind:timeType bind:profile {theme} />
		</Card>
	</Control>

	{#if !selectedItinerary && baseQuery && routingResponses.length !== 0}
		<Control position="top-left">
			<Card
				class="w-[500px] max-h-[70vh] overflow-y-auto overflow-x-hidden bg-background rounded-lg"
			>
				<ItineraryList {routingResponses} {baseQuery} bind:selectedItinerary />
			</Card>
		</Control>
	{/if}

	{#if selectedItinerary}
		<Control position="top-left">
			<Card class="w-[500px] bg-background rounded-lg">
				<div class="w-full flex justify-between items-center shadow-md pl-1 mb-1">
					<h2 class="ml-2 text-base font-semibold tracking-tight">Journey Details</h2>
					<Button
						variant="ghost"
						on:click={() => {
							selectedItinerary = undefined;
						}}
					>
						<X />
					</Button>
				</div>
				<div class="overflow-y-auto overflow-x-hidden max-h-[70vh]">
					<ConnectionDetail itinerary={selectedItinerary} />
				</div>
			</Card>
		</Control>
		<ItineraryGeoJson itinerary={selectedItinerary} {level} />
	{/if}
</Map>
