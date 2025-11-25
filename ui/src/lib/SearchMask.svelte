<script lang="ts">
	import { t } from '$lib/i18n/translation';
	import { ArrowUpDown, LocateFixed } from '@lucide/svelte';
	import maplibregl from 'maplibre-gl';
	import * as RadioGroup from '$lib/components/ui/radio-group';
	import Button from '$lib/components/ui/button/button.svelte';
	import { Label } from '$lib/components/ui/label';
	import type {
		ElevationCosts,
		Mode,
		PedestrianProfile,
		ServerConfig
	} from '@motis-project/motis-client';
	import AddressTypeahead from '$lib/AddressTypeahead.svelte';
	import AdvancedOptions from '$lib/AdvancedOptions.svelte';
	import DateInput from '$lib/DateInput.svelte';
	import { posToLocation, type Location } from '$lib/Location';
	import type { PrePostDirectMode } from '$lib/Modes';

	let {
		geocodingBiasPlace,
		serverConfig,
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
		maxPreTransitTime = $bindable(),
		maxPostTransitTime = $bindable(),
		maxDirectTime = $bindable(),
		ignorePreTransitRentalReturnConstraints = $bindable(),
		ignorePostTransitRentalReturnConstraints = $bindable(),
		ignoreDirectRentalReturnConstraints = $bindable(),
		preTransitProviderGroups = $bindable(),
		postTransitProviderGroups = $bindable(),
		directProviderGroups = $bindable()
	}: {
		geocodingBiasPlace?: maplibregl.LngLatLike;
		serverConfig: ServerConfig | undefined;
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
		maxPreTransitTime: number;
		maxPostTransitTime: number;
		maxDirectTime: number;
		ignorePreTransitRentalReturnConstraints: boolean;
		ignorePostTransitRentalReturnConstraints: boolean;
		ignoreDirectRentalReturnConstraints: boolean;
		preTransitProviderGroups: string[];
		postTransitProviderGroups: string[];
		directProviderGroups: string[];
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

<div id="searchmask-container" class="flex flex-col space-y-4 p-4 relative">
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
		class="absolute z-10 right-4 top-0"
		size="icon"
		onclick={() => getLocation()}
	>
		<LocateFixed class="w-5 h-5" />
	</Button>
	<Button
		class="absolute z-10 right-14 top-6"
		variant="outline"
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
	<div class="flex flex-row gap-2 flex-wrap">
		<DateInput bind:value={time} />
		<RadioGroup.Root
			class="flex"
			bind:value={() => (arriveBy ? 'arrival' : 'departure'), (v) => (arriveBy = v === 'arrival')}
		>
			<Label
				for="departure"
				class="flex items-center rounded-md border-2 border-muted bg-popover p-1 px-2 hover:bg-accent hover:text-accent-foreground [&:has([data-state=checked])]:border-blue-600 hover:cursor-pointer"
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
				class="flex items-center rounded-md border-2 border-muted bg-popover p-1 px-2 hover:bg-accent hover:text-accent-foreground [&:has([data-state=checked])]:border-blue-600 hover:cursor-pointer"
			>
				<RadioGroup.Item value="arrival" id="arrival" class="sr-only" aria-label={t.arrival} />
				<span>{t.arrival}</span>
			</Label>
		</RadioGroup.Root>
		<AdvancedOptions
			{serverConfig}
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
		/>
	</div>
</div>
