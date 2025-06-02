<script lang="ts">
	import AddressTypeahead from '$lib/AddressTypeahead.svelte';
	import Button from '$lib/components/ui/button/button.svelte';
	import { posToLocation, type Location } from '$lib/Location';
	import { t } from '$lib/i18n/translation';
	import * as Select from '$lib/components/ui/select';
	import { Label } from '$lib/components/ui/label';
	import * as RadioGroup from '$lib/components/ui/radio-group';
	import ChevronUp from 'lucide-svelte/icons/chevron-up';
	import ChevronDown from 'lucide-svelte/icons/chevron-down';
	import LocateFixed from 'lucide-svelte/icons/locate-fixed';
	import { Slider } from 'bits-ui';
	import { untrack } from 'svelte';
	import {
		oneToAll,
		plan,
		type OneToAllData,
		type OneToAllResponse,
		type ReachablePlace
	} from './api/openapi';
	import { lngLatToStr } from './lngLatToStr';
	import DateInput from './DateInput.svelte';
	import StreetModes from '$lib/StreetModes.svelte';
	import TransitModeSelect from '$lib/TransitModeSelect.svelte';
	import { prePostDirectModes, prePostModesToModes, type PrePostDirectMode, type TransitMode } from './Modes';
	import { formatDurationSec } from './formatDuration';
	import AdvancedOptions from './AdvancedOptions.svelte';

	interface IsochronesPos {
		lat: number;
		lng: number;
		seconds: number;
		name?: string;
	}
	const toPlaceString = (l: Location) => {
		if (l.match?.type === 'STOP') {
			return l.match.id;
		} else if (l.match?.level) {
			return `${lngLatToStr(l.match!)},${l.match.level}`;
		} else {
			return `${lngLatToStr(l.match!)},0`;
		}
	};
	const minutesToSeconds = (minutes: number[]) => { return minutes.map((m) => m * 60); }

	let {
		one = $bindable(),
		// maxTravelTime = $bindable(),
		geocodingBiasPlace,
		isochronesData = $bindable(),
		time = $bindable(),
		transitModes = $bindable(),
		arriveBy = $bindable(),
		color = $bindable(),
		opacity = $bindable()
	}: {
		one: Location;
		// maxTravelTime: string;
		geocodingBiasPlace?: maplibregl.LngLatLike;
		isochronesData: IsochronesPos[];
		time: Date;
		transitModes: TransitMode[];
		arriveBy: boolean;
		color: string;
		opacity: number;
	} = $props();

	const maxSupportedTransfers = 15;
	let maxTransfers = $state(maxSupportedTransfers);
	const possibleMaxTransfers = [...Array(maxSupportedTransfers).keys()
		.map((i) => i + 1)
		.map((i) => ({
			value: i.toString(),
			label: i.toString(),
		}))
	];
	const timeout = 60;

	const possibleMaxTravelTimes = minutesToSeconds([1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 75, 80, 90, 120, 150, 180, 210, 240])
		.map((s) => ({ value: s.toString(), label: formatDurationSec(s) }));
	;
	const possiblePrePostDurations = minutesToSeconds([1, 5, 10, 15, 20, 25, 30, 45, 60]);
	let expanded = $state<boolean>(false);

	let maxTravelTime = $state(45 * 60);
	let preTransitModes = $state<PrePostDirectMode[]>(['WALK']);
	let postTransitModes = $state<PrePostDirectMode[]>(['WALK']);
	let maxPreTransitTime = $state(15 * 60);
	let maxPostTransitTime = $state(15 * 60);
	let oneItems = $state<Array<Location>>([]);

	const ignorePreTransitRentalReturnConstraints = false;
	const ignorePostTransitRentalReturnConstraints = false;

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
						preTransitModes: arriveBy ? undefined : prePostModesToModes(preTransitModes),
						postTransitModes: arriveBy ? prePostModesToModes(postTransitModes) : undefined,
						maxPreTransitTime: arriveBy ? undefined : maxPreTransitTime,
						maxPostTransitTime: arriveBy ? maxPostTransitTime : undefined
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
	}

	const applyPosition = (position: { coords: { latitude: number; longitude: number } }) => {
		one = posToLocation({ lat: position.coords.latitude, lon: position.coords.longitude }, 0);
	};
</script>

<div id="isochrones-searchmask-container" class="flex flex-col space-y-4 p-4 relative">
	<AddressTypeahead
		place={geocodingBiasPlace}
		name="one"
		placeholder={t.from}
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
				<RadioGroup.Item value="arrival" id="isochrones-arrival" class="sr-only" aria-label={t.arrival} />
				<span>{t.arrival}</span>
			</Label>
		</RadioGroup.Root>
		<AdvancedOptions
			useRoutedTransfers={false}
			wheelchair={false}
			requireCarTransport={false}
			requireBikeTransport={false}
			bind:transitModes
			showTransitRestrictions={true}
			bind:maxTransfers
			bind:maxTravelTime
			{possibleMaxTransfers}
			{possibleMaxTravelTimes}
			bind:preTransitModes
			bind:postTransitModes
			directModes={[]}
			bind:maxPreTransitTime
			bind:maxPostTransitTime
			showDirectRestrictions={false}
			maxDirectTime={0}
			elevationCosts={'NONE'}
			ignorePreTransitRentalReturnConstraints
			ignorePostTransitRentalReturnConstraints
			ignoreDirectRentalReturnConstraints={false}
		/>
	</div>
</div>
