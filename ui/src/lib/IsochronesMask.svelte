<script lang="ts">
	import { t } from '$lib/i18n/translation';
	import { Slider } from 'bits-ui';
	import { LocateFixed } from '@lucide/svelte';
	import maplibregl from 'maplibre-gl';
	import * as RadioGroup from '$lib/components/ui/radio-group';
	import Button from '$lib/components/ui/button/button.svelte';
	import { Label } from '$lib/components/ui/label';
	import {
		type ElevationCosts,
		type PedestrianProfile,
		type ServerConfig
	} from '@motis-project/motis-client';
	import * as Select from '$lib/components/ui/select';
	import type { DisplayLevel, IsochronesOptions } from '$lib/map/IsochronesShared';
	import AddressTypeahead from '$lib/AddressTypeahead.svelte';
	import AdvancedOptions from '$lib/AdvancedOptions.svelte';
	import DateInput from '$lib/DateInput.svelte';
	import { posToLocation, type Location } from '$lib/Location';
	import { formatDurationSec } from '$lib/formatDuration';
	import type { PrePostDirectMode, TransitMode } from '$lib/Modes';
	import { generateTimes } from './generateTimes';

	let {
		one = $bindable(),
		maxTravelTime = $bindable(),
		serverConfig,
		geocodingBiasPlace,
		time = $bindable(),
		useRoutedTransfers = $bindable(),
		pedestrianProfile = $bindable(),
		requireBikeTransport = $bindable(),
		requireCarTransport = $bindable(),
		transitModes = $bindable(),
		maxTransfers = $bindable(),
		preTransitModes = $bindable(),
		postTransitModes = $bindable(),
		maxPreTransitTime = $bindable(),
		maxPostTransitTime = $bindable(),
		arriveBy = $bindable(),
		elevationCosts = $bindable(),
		ignorePreTransitRentalReturnConstraints = $bindable(),
		ignorePostTransitRentalReturnConstraints = $bindable(),
		options = $bindable(),
		preTransitProviderGroups = $bindable(),
		postTransitProviderGroups = $bindable(),
		directProviderGroups = $bindable()
	}: {
		one: Location;
		maxTravelTime: number;
		serverConfig: ServerConfig | undefined;
		geocodingBiasPlace?: maplibregl.LngLatLike;
		time: Date;
		useRoutedTransfers: boolean;
		pedestrianProfile: PedestrianProfile;
		requireBikeTransport: boolean;
		requireCarTransport: boolean;
		transitModes: TransitMode[];
		maxTransfers: number;
		preTransitModes: PrePostDirectMode[];
		postTransitModes: PrePostDirectMode[];
		maxPreTransitTime: number;
		maxPostTransitTime: number;
		arriveBy: boolean;
		elevationCosts: ElevationCosts;
		ignorePreTransitRentalReturnConstraints: boolean;
		ignorePostTransitRentalReturnConstraints: boolean;
		options: IsochronesOptions;
		preTransitProviderGroups: string[];
		postTransitProviderGroups: string[];
		directProviderGroups: string[];
	} = $props();
	const minutesToSeconds = (n: number): number => n * 60;
	const possibleMaxTravelTimes = $derived(
		generateTimes(
			minutesToSeconds(Math.min(serverConfig?.maxOneToAllTravelTimeLimit ?? 4 * 60, 6 * 60))
		).map((s) => ({
			value: s.toString(),
			label: formatDurationSec(s)
		}))
	);

	const displayLevels = new Map<DisplayLevel, string>([
		['OVERLAY_RECTS', t.isochrones.canvasRects],
		['OVERLAY_CIRCLES', t.isochrones.canvasCircles],
		['GEOMETRY_CIRCLES', t.isochrones.geojsonCircles]
	]);
	const possibleDisplayLevels = [
		...[...displayLevels.entries()].map(([id, label]) => ({ value: id, label: label }))
	];

	let oneItems = $state<Array<Location>>([]);

	let lastSearchDir = arriveBy ? 'arrival' : 'departure';

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
			const tmpProviderGroups = preTransitProviderGroups;
			preTransitProviderGroups = postTransitProviderGroups;
			postTransitProviderGroups = tmpProviderGroups;
			lastSearchDir = searchDir;
		}
	};

	const applyPosition = (position: { coords: { latitude: number; longitude: number } }) => {
		one = posToLocation({ lat: position.coords.latitude, lon: position.coords.longitude }, 0);
	};
</script>

{#snippet additionalComponents()}
	<div class="grid grid-cols-[2fr_2fr_1fr] items-center gap-2">
		<Select.Root type="single" bind:value={options.displayLevel}>
			<Select.Trigger class="overflow-hidden" aria-label={t.isochrones.displayLevel}>
				{displayLevels.get(options.displayLevel)}
			</Select.Trigger>
			<Select.Content sideOffset={10}>
				{#each possibleDisplayLevels as level, i (i + level.value)}
					<Select.Item value={level.value} label={level.label}>
						{level.label}
					</Select.Item>
				{/each}
			</Select.Content>
		</Select.Root>
		<Slider.Root
			type="single"
			min={0}
			max={1000}
			bind:value={options.opacity}
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
		<input class="flex right-0 align-right" type="color" bind:value={options.color} />
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
			{serverConfig}
			bind:wheelchair={
				() => pedestrianProfile === 'WHEELCHAIR',
				(v) => (pedestrianProfile = v ? 'WHEELCHAIR' : 'FOOT')
			}
			bind:requireCarTransport
			bind:requireBikeTransport
			bind:transitModes
			bind:maxTransfers
			bind:maxTravelTime
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
			bind:preTransitProviderGroups
			bind:postTransitProviderGroups
			bind:directProviderGroups
		/>
	</div>
</div>
