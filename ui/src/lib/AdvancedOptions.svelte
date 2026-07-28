<script lang="ts">
	import type { Snippet } from 'svelte';
	import { buttonVariants } from '$lib/components/ui/button';
	import * as Dialog from '$lib/components/ui/dialog';
	import { t } from '$lib/i18n/translation';
	import * as Select from '$lib/components/ui/select';
	import { Switch } from './components/ui/switch';
	import type {
		CyclingSpeed,
		ElevationCosts,
		PedestrianProfile,
		PedestrianSpeed,
		ServerConfig
	} from '@motis-project/motis-client';
	import { defaultQuery } from '$lib/defaults';
	import type { Location } from '$lib/Location';
	import { formatDurationSec } from './formatDuration';
	import {
		possibleTransitModes,
		prePostDirectModes,
		type PrePostDirectMode,
		type TransitMode
	} from './Modes';
	import NumberSelect from '$lib/NumberSelect.svelte';
	import StreetModes from '$lib/StreetModes.svelte';
	import TransitModeSelect from '$lib/TransitModeSelect.svelte';
	import { type NumberSelectOption } from '$lib/NumberSelect.svelte';
	import { generateTimes } from './generateTimes';
	import ViaStopOptions from './ViaStopOptions.svelte';
	import Slider from './components/ui/slider/Slider.svelte';
	let {
		advancedOptionsOpen = $bindable(false),
		useRoutedTransfers = $bindable(),
		serverConfig,
		wheelchair = $bindable(),
		requireBikeTransport = $bindable(),
		requireCarTransport = $bindable(),
		transitModes = $bindable(),
		maxTransfers = $bindable(),
		maxTravelTime = $bindable(undefined),
		possibleMaxTravelTimes = [],
		preTransitModes = $bindable(),
		postTransitModes = $bindable(),
		directModes = $bindable(undefined),
		transferTimeFactor = $bindable(),
		maxPreTransitTime = $bindable(),
		maxPostTransitTime = $bindable(),
		maxDirectTime = $bindable(undefined),
		elevationCosts = $bindable(),
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
		pedestrianProfile = $bindable(),
		hasDebug = false,
		additionalComponents
	}: {
		advancedOptionsOpen: boolean;
		useRoutedTransfers: boolean;
		serverConfig: ServerConfig | undefined;
		wheelchair: boolean;
		requireBikeTransport: boolean;
		requireCarTransport: boolean;
		transitModes: TransitMode[];
		maxTransfers: number;
		maxTravelTime: number | undefined;
		possibleMaxTravelTimes?: NumberSelectOption[];
		preTransitModes: PrePostDirectMode[];
		postTransitModes: PrePostDirectMode[];
		directModes: PrePostDirectMode[] | undefined;
		transferTimeFactor: number;
		maxPreTransitTime: number;
		maxPostTransitTime: number;
		maxDirectTime: number | undefined;
		elevationCosts: ElevationCosts;
		ignorePreTransitRentalReturnConstraints: boolean;
		ignorePostTransitRentalReturnConstraints: boolean;
		ignoreDirectRentalReturnConstraints: boolean | undefined;
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
		additionalComponents?: Snippet;
		pedestrianProfile: PedestrianProfile;
	} = $props();
	const possibleMaxTransfers = [...Array(defaultQuery.maxTransfers + 1).keys()].map((i) => ({
		value: i.toString(),
		label: i.toString()
	}));

	const possibleDirectDurations = $derived(
		generateTimes(serverConfig?.maxDirectTimeLimit ?? 60 * 60)
	);
	const possiblePrePostDurations = $derived(
		generateTimes(serverConfig?.maxPrePostTransitTimeLimit ?? 60 * 60)
	);
	const hasBikeMode = $derived(
		preTransitModes.includes('BIKE') ||
			postTransitModes.includes('BIKE') ||
			directModes?.includes('BIKE')
	);

	function setModes(mode: PrePostDirectMode) {
		return function (checked: boolean) {
			if (checked) {
				preTransitModes = [mode];
				postTransitModes = [mode];
				directModes = [mode];
			}
		};
	}
	if (transitModes.includes('TRANSIT')) {
		transitModes = [...possibleTransitModes];
	}
	if (requireBikeTransport) {
		setModes('BIKE')(true);
	}
	if (requireCarTransport) {
		setModes('CAR')(true);
	}

	const possibleElevationCosts = [
		{ value: 'NONE' as ElevationCosts, label: t.elevationCosts.NONE },
		{ value: 'LOW' as ElevationCosts, label: t.elevationCosts.LOW },
		{ value: 'HIGH' as ElevationCosts, label: t.elevationCosts.HIGH }
	];
	let allowElevationCosts = $derived(
		serverConfig?.hasElevation &&
			(requireBikeTransport ||
				preTransitModes.includes('BIKE') ||
				postTransitModes.includes('BIKE') ||
				directModes?.includes('BIKE'))
	);
	let allowStreetRouting = $derived(serverConfig?.hasStreetRouting);
	let allowRoutedTransfers = $derived(serverConfig?.hasRoutedTransfers);
	$effect(() => {
		transferTimeFactor = Math.max(1, defaultQuery.pedestrianSpeed / pedestrianSpeed);
	});
	let showHgvOptions = $derived(
		preTransitModes.includes('HGV') ||
			postTransitModes.includes('HGV') ||
			directModes?.includes('HGV')
	);

	let possibleModes = $derived(
		hasDebug ? prePostDirectModes : prePostDirectModes.filter((m) => !m.startsWith('DEBUG_'))
	);

	const inputClass =
		'flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm transition-colors placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50';
</script>

{#snippet optionsContent()}
	<TransitModeSelect bind:transitModes />

	<div class="space-y-2">
		<div class="grid grid-cols-2">
			<Switch
				bind:checked={useRoutedTransfers}
				disabled={!allowRoutedTransfers}
				label={t.useRoutedTransfers}
				id="useRoutedTransfers"
				onCheckedChange={(checked) => {
					if (wheelchair && !checked) {
						wheelchair = false;
					}
				}}
			/>

			<Switch
				bind:checked={wheelchair}
				disabled={!allowRoutedTransfers}
				label={t.wheelchair}
				id="wheelchair"
				onCheckedChange={(checked) => {
					if (checked && !useRoutedTransfers) {
						useRoutedTransfers = true;
					}
				}}
			/>
			<Switch
				bind:checked={requireBikeTransport}
				label={t.requireBikeTransport}
				onCheckedChange={setModes('BIKE')}
				id="requireBikeTransport"
			/>
			<Switch
				bind:checked={requireCarTransport}
				disabled={!allowRoutedTransfers}
				label={t.requireCarTransport}
				id="requireCarTransport"
				onCheckedChange={(checked) => {
					if (checked && !useRoutedTransfers && allowRoutedTransfers) {
						useRoutedTransfers = true;
					}
					setModes('CAR')(checked);
				}}
			/>
		</div>
		<ViaStopOptions bind:via bind:viaMinimumStay bind:viaLabels />

		<div
			class="grid grid-cols-4
				items-center gap-x-4 gap-y-2"
		>
			<div class="text-sm">
				{t.routingSegments.maxTransfers}
			</div>
			<NumberSelect bind:value={maxTransfers} possibleValues={possibleMaxTransfers} />
			{#if maxTravelTime !== undefined}
				<div class="text-sm">
					{t.routingSegments.maxTravelTime}
				</div>
				<NumberSelect
					bind:value={maxTravelTime}
					possibleValues={possibleMaxTravelTimes}
					labelFormatter={formatDurationSec}
				/>
			{/if}
			<div class="text-sm">{t.routingSegments.additionalTransferTime}</div>
			<input
				type="number"
				min="0"
				bind:value={additionalTransferTime}
				placeholder={t.duration + ' (min)'}
				class="text-sm border w-full h-full pl-1 text-center rounded-md"
			/>
		</div>
		<!-- First mile -->
		<StreetModes
			label={t.routingSegments.firstMile}
			disabled={!allowStreetRouting}
			bind:modes={preTransitModes}
			bind:maxTransitTime={maxPreTransitTime}
			{possibleModes}
			possibleMaxTransitTime={possiblePrePostDurations}
			bind:ignoreRentalReturnConstraints={ignorePreTransitRentalReturnConstraints}
			bind:providerGroups={preTransitProviderGroups}
		></StreetModes>

		<!-- Last mile -->
		<StreetModes
			label={t.routingSegments.lastMile}
			disabled={!allowStreetRouting}
			bind:modes={postTransitModes}
			bind:maxTransitTime={maxPostTransitTime}
			{possibleModes}
			possibleMaxTransitTime={possiblePrePostDurations}
			bind:ignoreRentalReturnConstraints={ignorePostTransitRentalReturnConstraints}
			bind:providerGroups={postTransitProviderGroups}
		></StreetModes>

		<!-- Direct -->
		{#if directModes !== undefined && maxDirectTime !== undefined && ignoreDirectRentalReturnConstraints !== undefined}
			<StreetModes
				label={t.routingSegments.direct}
				disabled={!allowStreetRouting}
				bind:modes={directModes}
				bind:maxTransitTime={maxDirectTime}
				{possibleModes}
				possibleMaxTransitTime={possibleDirectDurations}
				bind:ignoreRentalReturnConstraints={ignoreDirectRentalReturnConstraints}
				bind:providerGroups={directProviderGroups}
			></StreetModes>
		{/if}

		{#if showHgvOptions}
			<div class="space-y-2">
				<div class="text-sm font-medium">{t.hgvRoutingOptions}</div>
				<div class="grid grid-cols-2 items-center gap-2">
					<div class="text-sm">{t.vehicleHeight}</div>
					<input
						class={inputClass}
						disabled={!allowStreetRouting}
						type="number"
						min="0"
						step="0.01"
						bind:value={vehicleHeight}
					/>

					<div class="text-sm">{t.vehicleWidth}</div>
					<input
						class={inputClass}
						disabled={!allowStreetRouting}
						type="number"
						min="0"
						step="0.01"
						bind:value={vehicleWidth}
					/>

					<div class="text-sm">{t.vehicleLength}</div>
					<input
						class={inputClass}
						disabled={!allowStreetRouting}
						type="number"
						min="0"
						step="0.01"
						bind:value={vehicleLength}
					/>

					<div class="text-sm">{t.vehicleWeight}</div>
					<input
						class={inputClass}
						disabled={!allowStreetRouting}
						type="number"
						min="0"
						step="0.01"
						bind:value={vehicleWeight}
					/>

					<div class="text-sm">{t.vehicleTopSpeed}</div>
					<input
						class={inputClass}
						disabled={!allowStreetRouting}
						type="number"
						min="0"
						step="1"
						bind:value={vehicleTopSpeed}
					/>

					<div class="text-sm">{t.vehicleAxleCount}</div>
					<input
						class={inputClass}
						disabled={!allowStreetRouting}
						type="number"
						min="1"
						step="1"
						bind:value={vehicleAxleCount}
					/>

					<div class="text-sm">{t.vehicleAxleLoad}</div>
					<input
						class={inputClass}
						disabled={!allowStreetRouting}
						type="number"
						min="0"
						step="0.01"
						bind:value={vehicleAxleLoad}
					/>
				</div>
				<Switch
					disabled={!allowStreetRouting}
					bind:checked={vehicleHazmat}
					label={t.vehicleHazmat}
					id="vehicleHazmat"
					onCheckedChange={(checked) => {
						if (!checked) {
							vehicleHazmatWater = false;
						}
					}}
				/>
				<Switch
					disabled={!allowStreetRouting}
					bind:checked={vehicleHazmatWater}
					label={t.vehicleHazmatWater}
					id="vehicleHazmatWater"
					onCheckedChange={(checked) => {
						if (checked) {
							vehicleHazmat = true;
						}
					}}
				/>
				<Switch
					disabled={!allowStreetRouting}
					bind:checked={vehicleTrailer}
					label={t.vehicleTrailer}
					id="vehicleTrailer"
				/>
				<Switch
					disabled={!allowStreetRouting}
					bind:checked={vehicleLezAccess}
					label={t.vehicleLezAccess}
					id="vehicleLezAccess"
				/>
			</div>
		{/if}

		<div class="grid grid-cols-[1fr_2fr_1fr] text-sm items-center gap-2">
			<span>{t.routingSegments.pedestrianSpeed}</span>
			<Slider
				min={pedestrianProfile == 'FOOT' ? 0.8 : 0.5}
				step={0.1}
				max={pedestrianProfile == 'FOOT' ? 3 : 1.6}
				bind:value={pedestrianSpeed}
			/>
			<div>{(pedestrianSpeed * 3.6).toFixed(1)} km/h</div>
			<span>{t.routingSegments.cyclingSpeed}</span>
			<Slider min={2.7} max={7} step={0.1} disabled={!hasBikeMode} bind:value={cyclingSpeed} />
			<div>{(cyclingSpeed * 3.6).toFixed(1)} km/h</div>
		</div>
	</div>

	<!-- Elevation Costs -->
	<div class="grid grid-cols-2 items-center">
		<div class="text-sm">
			{t.selectElevationCosts}
		</div>
		<Select.Root
			disabled={!allowElevationCosts || !allowStreetRouting}
			type="single"
			bind:value={elevationCosts}
		>
			<Select.Trigger aria-label={t.selectElevationCosts}>
				{t.elevationCosts[elevationCosts]}
			</Select.Trigger>
			<Select.Content sideOffset={10}>
				{#each possibleElevationCosts as costs, i (i + costs.value)}
					<Select.Item value={costs.value} label={costs.label}>
						{costs.label}
					</Select.Item>
				{/each}
			</Select.Content>
		</Select.Root>
	</div>
	{#if additionalComponents}
		{@render additionalComponents()}
	{/if}
{/snippet}

<Dialog.Root bind:open={advancedOptionsOpen}>
	<Dialog.Trigger class={buttonVariants({ variant: 'ghost' })}>
		{t.advancedSearchOptions}
	</Dialog.Trigger>
	<Dialog.Content class="flex max-h-[90vh] max-w-2xl flex-col">
		<Dialog.Header>
			<Dialog.Title>{t.advancedSearchOptions}</Dialog.Title>
		</Dialog.Header>
		<div class="space-y-4 overflow-y-auto p-2">
			{@render optionsContent()}
		</div>
	</Dialog.Content>
</Dialog.Root>
