<script lang="ts">
	import { t } from '$lib/i18n/translation';
	import { ArrowUpDown, LocateFixed } from '@lucide/svelte';
	import maplibregl from 'maplibre-gl';
	import * as RadioGroup from '$lib/components/ui/radio-group';
	import Button from '$lib/components/ui/button/button.svelte';
	import { Label } from '$lib/components/ui/label';
	import {
		type CyclingSpeed,
		type PedestrianSpeed,
		type ElevationCosts,
		type Mode,
		type PedestrianProfile,
		type ServerConfig
	} from '@motis-project/motis-client';
	import AddressTypeahead from '$lib/AddressTypeahead.svelte';
	import AdvancedOptions from '$lib/AdvancedOptions.svelte';
	import DateInput from '$lib/DateInput.svelte';
	import { posToLocation, type Location } from '$lib/Location';
	import type { PrePostDirectMode } from '$lib/Modes';

	let {
		geocodingBiasPlace,
		serverConfig,
		advancedOptionsOpen = $bindable(),
		from = $bindable(),
		to = $bindable(),
		time = $bindable(),
		arriveBy = $bindable(),
		pedestrianProfile = $bindable(),
		useRoutedTransfers = $bindable(),
		maxTransfers = $bindable(),
		requireCarTransport = $bindable(),
		requireBikeTransport = $bindable(),
		transitModes = $bindable(),
		preTransitModes = $bindable(),
		postTransitModes = $bindable(),
		directModes = $bindable(),
		elevationCosts = $bindable(),
		transferTimeFactor = $bindable(),
		maxPreTransitTime = $bindable(),
		maxPostTransitTime = $bindable(),
		maxDirectTime = $bindable(),
		ignorePreTransitRentalReturnConstraints = $bindable(),
		ignorePostTransitRentalReturnConstraints = $bindable(),
		ignoreDirectRentalReturnConstraints = $bindable(),
		preTransitProviderGroups = $bindable(),
		postTransitProviderGroups = $bindable(),
		directProviderGroups = $bindable(),
		vehicleHeight = $bindable(),
		vehicleWidth = $bindable(),
		vehicleLength = $bindable(),
		vehicleWeight = $bindable(),
		vehicleHazmat = $bindable(),
		vehicleHazmatWater = $bindable(),
		vehicleAxleCount = $bindable(),
		vehicleAxleLoad = $bindable(),
		vehicleTrailer = $bindable(),
		vehicleTopSpeed = $bindable(),
		vehicleLezAccess = $bindable(),
		via = $bindable(),
		viaMinimumStay = $bindable(),
		viaLabels = $bindable(),
		pedestrianSpeed = $bindable(),
		cyclingSpeed = $bindable(),
		additionalTransferTime = $bindable(),
		hasDebug = false
	}: {
		geocodingBiasPlace?: maplibregl.LngLatLike;
		serverConfig: ServerConfig | undefined;
		advancedOptionsOpen: boolean;
		from: Location;
		to: Location;
		time: Date;
		arriveBy: boolean;
		pedestrianProfile: PedestrianProfile;
		useRoutedTransfers: boolean;
		maxTransfers: number;
		requireCarTransport: boolean;
		requireBikeTransport: boolean;
		transitModes: Mode[];
		preTransitModes: PrePostDirectMode[];
		postTransitModes: PrePostDirectMode[];
		directModes: PrePostDirectMode[];
		elevationCosts: ElevationCosts;
		transferTimeFactor: number;
		maxPreTransitTime: number;
		maxPostTransitTime: number;
		maxDirectTime: number;
		ignorePreTransitRentalReturnConstraints: boolean;
		ignorePostTransitRentalReturnConstraints: boolean;
		ignoreDirectRentalReturnConstraints: boolean;
		preTransitProviderGroups: string[];
		postTransitProviderGroups: string[];
		directProviderGroups: string[];
		vehicleHeight: number;
		vehicleWidth: number;
		vehicleLength: number;
		vehicleWeight: number;
		vehicleHazmat: boolean;
		vehicleHazmatWater: boolean;
		vehicleAxleCount: number;
		vehicleAxleLoad: number;
		vehicleTrailer: boolean;
		vehicleTopSpeed: number;
		vehicleLezAccess: boolean;
		via: undefined | Location[];
		viaMinimumStay: undefined | number[];
		viaLabels: Record<string, string>;
		pedestrianSpeed: PedestrianSpeed;
		cyclingSpeed: CyclingSpeed;
		additionalTransferTime: number | undefined;
		hasDebug: boolean;
	} = $props();

	let fromItems = $state<Array<Location>>([]);
	let toItems = $state<Array<Location>>([]);
	const getLocation = () => {
		if (navigator && navigator.geolocation) {
			navigator.geolocation.getCurrentPosition(applyPosition, (e) => console.log(e), {
				enableHighAccuracy: true
			});
		}
	};

	const applyPosition = (position: { coords: { latitude: number; longitude: number } }) => {
		from = posToLocation({ lat: position.coords.latitude, lon: position.coords.longitude }, 0);
	};
</script>

<div
	id="searchmask-container"
	class="flex max-h-full min-h-0 flex-col space-y-4 overflow-hidden p-4 relative"
>
	<AddressTypeahead
		place={geocodingBiasPlace}
		name="from"
		placeholder={t.from}
		bind:selected={from}
		bind:items={fromItems}
		{transitModes}
	/>
	<AddressTypeahead
		place={geocodingBiasPlace}
		name="to"
		placeholder={t.to}
		bind:selected={to}
		bind:items={toItems}
		{transitModes}
	/>
	<Button
		variant="ghost"
		title={t.myLocation}
		class="absolute z-10 right-4 top-0"
		size="icon"
		onclick={() => getLocation()}
	>
		<LocateFixed class="w-5 h-5" />
	</Button>
	<Button
		class="absolute z-10 right-14 top-6"
		variant="outline"
		title={t.reverseDirections}
		size="icon"
		onclick={() => {
			const tmp = to;
			to = from;
			from = tmp;

			const tmpItems = toItems;
			toItems = fromItems;
			fromItems = tmpItems;
		}}
	>
		<ArrowUpDown class="w-5 h-5" />
	</Button>
	<div class="flex min-h-0 flex-row gap-2 flex-wrap">
		<DateInput bind:value={time} />
		<RadioGroup.Root
			class="flex gap-0 grow"
			bind:value={() => (arriveBy ? 'arrival' : 'departure'), (v) => (arriveBy = v === 'arrival')}
		>
			<Label
				for="departure"
				class="flex grow justify-center items-center border-input rounded-l-md px-2 py-1 bg-accent text-gray-500 hover:bg-blue-100 [&:has([data-state=checked])]:bg-blue-600 [&:has([data-state=checked])]:text-white hover:cursor-pointer"
			>
				<RadioGroup.Item
					value="departure"
					id="departure"
					class="sr-only"
					aria-label={t.departure}
				/>
				<span>{t.departure}</span>
			</Label>
			<Label
				for="arrival"
				class="flex grow justify-center items-center border-input rounded-r-md px-2 py-1 bg-accent text-gray-500 hover:bg-blue-100 [&:has([data-state=checked])]:bg-blue-600 [&:has([data-state=checked])]:text-white hover:cursor-pointer"
			>
				<RadioGroup.Item value="arrival" id="arrival" class="sr-only" aria-label={t.arrival} />
				<span>{t.arrival}</span>
			</Label>
		</RadioGroup.Root>
		<AdvancedOptions
			{serverConfig}
			bind:advancedOptionsOpen
			bind:useRoutedTransfers
			bind:wheelchair={
				() => pedestrianProfile === 'WHEELCHAIR',
				(v) => (pedestrianProfile = v ? 'WHEELCHAIR' : 'FOOT')
			}
			bind:requireCarTransport
			bind:maxTransfers
			maxTravelTime={undefined}
			bind:requireBikeTransport
			bind:transitModes
			bind:preTransitModes
			bind:postTransitModes
			bind:directModes
			bind:transferTimeFactor
			bind:maxPreTransitTime
			bind:maxPostTransitTime
			bind:maxDirectTime
			bind:elevationCosts
			bind:ignorePreTransitRentalReturnConstraints
			bind:ignorePostTransitRentalReturnConstraints
			bind:ignoreDirectRentalReturnConstraints
			bind:preTransitProviderGroups
			bind:postTransitProviderGroups
			bind:directProviderGroups
			bind:vehicleHeight
			bind:vehicleWidth
			bind:vehicleLength
			bind:vehicleWeight
			bind:vehicleHazmat
			bind:vehicleHazmatWater
			bind:vehicleAxleCount
			bind:vehicleAxleLoad
			bind:vehicleTrailer
			bind:vehicleTopSpeed
			bind:vehicleLezAccess
			bind:via
			bind:viaMinimumStay
			bind:viaLabels
			bind:pedestrianSpeed
			bind:cyclingSpeed
			bind:additionalTransferTime
			bind:pedestrianProfile
			{hasDebug}
		/>
		<Button
			class="flex grow bg-blue-600 hover:!bg-blue-700 text-white font-bold"
			variant="default"
			title={t.search}
			onclick={() => {}}
		>
			{t.search}
		</Button>
	</div>
</div>
