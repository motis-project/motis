<script lang="ts">
	import { t } from '$lib/i18n/translation';
	import { untrack } from 'svelte';
	import { Slider } from 'bits-ui';
	import LocateFixed from 'lucide-svelte/icons/locate-fixed';
	import maplibregl from 'maplibre-gl';
	import * as RadioGroup from '$lib/components/ui/radio-group';
	import Button from '$lib/components/ui/button/button.svelte';
	import { Label } from '$lib/components/ui/label';
	import {
		oneToAll,
		type ElevationCosts,
		type OneToAllData,
		type OneToAllResponse,
		type PedestrianProfile,
		type ReachablePlace
	} from '$lib/api/openapi';
	import AddressTypeahead from '$lib/AddressTypeahead.svelte';
	import AdvancedOptions from '$lib/AdvancedOptions.svelte';
	import DateInput from '$lib/DateInput.svelte';
	import { posToLocation, type Location } from '$lib/Location';
	import { formatDurationSec } from '$lib/formatDuration';
	import { lngLatToStr } from '$lib/lngLatToStr';
	import { prePostModesToModes, type PrePostDirectMode, type TransitMode } from '$lib/Modes';
	import { type IsochronesPos } from '$lib/map/Isochrones.svelte';

	const toPlaceString = (l: Location) => {
		if (l.match?.type === 'STOP') {
			return l.match.id;
		} else if (l.match?.level) {
			return `${lngLatToStr(l.match!)},${l.match.level}`;
		} else {
			return `${lngLatToStr(l.match!)},0`;
		}
	};
	const minutesToSeconds = (minutes: number[]) => {
		return minutes.map((m) => m * 60);
	};

	let {
		one = $bindable(),
		maxTravelTime = $bindable(),
		geocodingBiasPlace,
		isochronesData = $bindable(),
		time = $bindable(),
		useRoutedTransfers = $bindable(),
		pedestrianProfile = $bindable(),
		requireBikeTransport = $bindable(),
		requireCarTransport = $bindable(),
		transitModes = $bindable(),
		preTransitModes = $bindable(),
		postTransitModes = $bindable(),
		maxPreTransitTime = $bindable(),
		maxPostTransitTime = $bindable(),
		arriveBy = $bindable(),
		elevationCosts = $bindable(),
		ignorePreTransitRentalReturnConstraints = $bindable(),
		ignorePostTransitRentalReturnConstraints = $bindable(),
		color = $bindable(),
		opacity = $bindable()
	}: {
		one: Location;
		maxTravelTime: number;
		geocodingBiasPlace?: maplibregl.LngLatLike;
		isochronesData: IsochronesPos[];
		time: Date;
		useRoutedTransfers: boolean;
		pedestrianProfile: PedestrianProfile;
		requireBikeTransport: boolean;
		requireCarTransport: boolean;
		transitModes: TransitMode[];
		preTransitModes: PrePostDirectMode[];
		postTransitModes: PrePostDirectMode[];
		maxPreTransitTime: number;
		maxPostTransitTime: number;
		arriveBy: boolean;
		elevationCosts: ElevationCosts;
		ignorePreTransitRentalReturnConstraints: boolean;
		ignorePostTransitRentalReturnConstraints: boolean;
		color: string;
		opacity: number;
	} = $props();

	const maxSupportedTransfers = 15;
	let maxTransfers = $state(maxSupportedTransfers);
	const possibleMaxTransfers = [
		...Array(maxSupportedTransfers)
			.keys()
			.map((i) => i + 1)
			.map((i) => ({
				value: i.toString(),
				label: i.toString()
			}))
	];
	const timeout = 60;

	const possibleMaxTravelTimes = minutesToSeconds([
		1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 75, 80, 90, 120, 150, 180, 210, 240
	]).map((s) => ({ value: s.toString(), label: formatDurationSec(s) }));

	let oneItems = $state<Array<Location>>([]);

	let lastSearchDir = arriveBy ? 'arrival' : 'departure';

	let queryTimeout: number;

	let isochronesQuery = $derived(
		one?.match
			? ({
					query: {
						one: toPlaceString(one),
						maxTravelTime: Math.ceil(maxTravelTime / 60),
						time: time.toISOString(),
						transitModes,
						maxTransfers,
						arriveBy,
						useRoutedTransfers,
						wheelchair: pedestrianProfile === 'WHEELCHAIR',
						requireBikeTransport,
						requireCarTransport,
						preTransitModes: arriveBy ? undefined : prePostModesToModes(preTransitModes),
						postTransitModes: arriveBy ? prePostModesToModes(postTransitModes) : undefined,
						maxPreTransitTime: arriveBy ? undefined : maxPreTransitTime,
						maxPostTransitTime: arriveBy ? maxPostTransitTime : undefined,
						elevationCosts,
						maxMatchingDistance: pedestrianProfile == 'WHEELCHAIR' ? 8 : 250,
						ignorePreTransitRentalReturnConstraints,
						ignorePostTransitRentalReturnConstraints
					}
				} as OneToAllData)
			: undefined
	);
	$effect(() => {
		if (isochronesQuery) {
			clearTimeout(queryTimeout);
			queryTimeout = setTimeout(() => {
				oneToAll(isochronesQuery).then(
					(r: { data: OneToAllResponse | undefined; error: unknown }) => {
						if (r.error) {
							throw new Error(String(r.error));
						}
						const all = r.data!.all!.map((p: ReachablePlace) => {
							return {
								lat: p.place?.lat,
								lng: p.place?.lon,
								seconds: maxTravelTime - 60 * (p.duration ?? 0),
								name: p.place?.name
							} as IsochronesPos;
						});
						untrack(() => {
							isochronesData = [...all];
						});
					}
				);
			}, timeout);
		}
	});

	const getLocation = () => {
		if (navigator && navigator.geolocation) {
			navigator.geolocation.getCurrentPosition(applyPosition, (e) => console.log(e), {
				enableHighAccuracy: true
			});
		}
	};
	const swapPrePostData = (searchDir: string) => {
		if (searchDir != lastSearchDir) {
			const tmpModes = preTransitModes;
			preTransitModes = postTransitModes;
			postTransitModes = tmpModes;
			const tmpTime = maxPreTransitTime;
			maxPreTransitTime = maxPostTransitTime;
			maxPostTransitTime = tmpTime;
			lastSearchDir = searchDir;
		}
	};

	const applyPosition = (position: { coords: { latitude: number; longitude: number } }) => {
		one = posToLocation({ lat: position.coords.latitude, lon: position.coords.longitude }, 0);
	};
</script>

{#snippet additionalComponents()}
	<div class="grid grid-cols-[1fr_2fr_1fr] items-center gap-2">
		<div class="text-sm">
			{t.isochronesStyling}
		</div>
		<Slider.Root
			type="single"
			min={0}
			max={1000}
			bind:value={opacity}
			class="relative flex w-full touch-none select-none items-center"
		>
			<span class="bg-dark-10 relative h-2 w-full grow cursor-pointer overflow-hidden rounded-full">
				<Slider.Range class="bg-foreground absolute h-full" />
			</span>
			<Slider.Thumb
				index={0}
				class="border-border-input bg-background hover:border-dark-40 focus-visible:ring-foreground dark:bg-foreground dark:shadow-card focus-visible:outline-hidden block size-[25px] cursor-pointer rounded-full border shadow-sm transition-colors focus-visible:ring-2 focus-visible:ring-offset-2 active:scale-[0.98] disabled:pointer-events-none disabled:opacity-50"
			/>
		</Slider.Root>
		<input class="flex right-0 align-right" type="color" bind:value={color} />
	</div>
{/snippet}

<div id="isochrones-searchmask-container" class="flex flex-col space-y-4 p-4 relative">
	<AddressTypeahead
		place={geocodingBiasPlace}
		name="one"
		placeholder={t.position}
		bind:selected={one}
		bind:items={oneItems}
	/>
	<Button
		variant="ghost"
		class="absolute z-10 right-4 top-0"
		size="icon"
		onclick={() => getLocation()}
	>
		<LocateFixed class="w-5 h-5" />
	</Button>
	<div class="flex flex-row gap-2 flex-wrap">
		<DateInput bind:value={time} />
		<RadioGroup.Root
			class="flex"
			bind:value={() => (arriveBy ? 'arrival' : 'departure'), (v) => (arriveBy = v === 'arrival')}
			onValueChange={swapPrePostData}
		>
			<Label
				for="isochrones-departure"
				class="flex items-center rounded-md border-2 border-muted bg-popover p-1 px-2 hover:bg-accent hover:text-accent-foreground [&:has([data-state=checked])]:border-blue-600 hover:cursor-pointer"
			>
				<RadioGroup.Item
					value="departure"
					id="isochrones-departure"
					class="sr-only"
					aria-label={t.departure}
				/>
				<span>{t.departure}</span>
			</Label>
			<Label
				for="isochrones-arrival"
				class="flex items-center rounded-md border-2 border-muted bg-popover p-1 px-2 hover:bg-accent hover:text-accent-foreground [&:has([data-state=checked])]:border-blue-600 hover:cursor-pointer"
			>
				<RadioGroup.Item
					value="arrival"
					id="isochrones-arrival"
					class="sr-only"
					aria-label={t.arrival}
				/>
				<span>{t.arrival}</span>
			</Label>
		</RadioGroup.Root>
		<AdvancedOptions
			bind:useRoutedTransfers
			bind:wheelchair={
				() => pedestrianProfile === 'WHEELCHAIR',
				(v) => (pedestrianProfile = v ? 'WHEELCHAIR' : 'FOOT')
			}
			bind:requireCarTransport
			bind:requireBikeTransport
			bind:transitModes
			bind:maxTransfers
			bind:maxTravelTime
			{possibleMaxTransfers}
			{possibleMaxTravelTimes}
			bind:preTransitModes
			bind:postTransitModes
			directModes={undefined}
			bind:maxPreTransitTime
			bind:maxPostTransitTime
			maxDirectTime={undefined}
			bind:elevationCosts
			bind:ignorePreTransitRentalReturnConstraints
			bind:ignorePostTransitRentalReturnConstraints
			ignoreDirectRentalReturnConstraints={undefined}
			{additionalComponents}
		/>
	</div>
</div>
