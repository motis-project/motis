<script lang="ts">
	import type { Snippet } from 'svelte';
	import Button from '$lib/components/ui/button/button.svelte';
	import { t } from '$lib/i18n/translation';
	import * as Select from '$lib/components/ui/select';
	import ChevronUp from 'lucide-svelte/icons/chevron-up';
	import ChevronDown from 'lucide-svelte/icons/chevron-down';
	import { Switch } from './components/ui/switch';
	import type { ElevationCosts } from '$lib/api/openapi';
	import { defaultQuery } from '$lib/defaults';
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

	let {
		useRoutedTransfers = $bindable(),
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
		maxPreTransitTime = $bindable(),
		maxPostTransitTime = $bindable(),
		maxDirectTime = $bindable(undefined),
		elevationCosts = $bindable(),
		ignorePreTransitRentalReturnConstraints = $bindable(),
		ignorePostTransitRentalReturnConstraints = $bindable(),
		ignoreDirectRentalReturnConstraints = $bindable(),
		additionalComponents
	}: {
		useRoutedTransfers: boolean;
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
		maxPreTransitTime: number;
		maxPostTransitTime: number;
		maxDirectTime: number | undefined;
		elevationCosts: ElevationCosts;
		ignorePreTransitRentalReturnConstraints: boolean;
		ignorePostTransitRentalReturnConstraints: boolean;
		ignoreDirectRentalReturnConstraints: boolean | undefined;
		additionalComponents?: Snippet;
	} = $props();

	const possibleMaxTransfers = [...Array(defaultQuery.maxTransfers + 1).keys()].map((i) => ({
		value: i.toString(),
		label: i.toString()
	}));

	const possibleDirectDurations = [
		5 * 60,
		10 * 60,
		15 * 60,
		20 * 60,
		25 * 60,
		30 * 60,
		35 * 60,
		40 * 60,
		45 * 60,
		50 * 60,
		60 * 60,
		90 * 60,
		120 * 60,
		180 * 60,
		240 * 60,
		300 * 60,
		360 * 60
	];
	const possiblePrePostDurations = [
		1 * 60,
		5 * 60,
		10 * 60,
		15 * 60,
		20 * 60,
		25 * 60,
		30 * 60,
		40 * 60,
		50 * 60,
		60 * 60
	];
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
	let expanded = $state<boolean>(false);
	let allowElevationCosts = $derived(
		requireBikeTransport ||
			preTransitModes.includes('BIKE') ||
			postTransitModes.includes('BIKE') ||
			directModes?.includes('BIKE')
	);
</script>

<Button variant="ghost" onclick={() => (expanded = !expanded)}>
	{t.advancedSearchOptions}
	{#if expanded}
		<ChevronUp class="size-[18px]" />
	{:else}
		<ChevronDown class="size-[18px]" />
	{/if}
</Button>

{#if expanded}
	<div class="w-full space-y-4">
		<TransitModeSelect bind:transitModes />

		<div class="space-y-2">
			<Switch
				bind:checked={useRoutedTransfers}
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
				label={t.requireCarTransport}
				id="requireCarTransport"
				onCheckedChange={(checked) => {
					if (checked && !useRoutedTransfers) {
						useRoutedTransfers = true;
					}
					setModes('CAR')(checked);
				}}
			/>

			<div
				class="grid {maxTravelTime === undefined
					? 'grid-cols-2'
					: 'grid-cols-4'} items-center gap-2"
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
			</div>

			<!-- First mile -->
			<StreetModes
				label={t.routingSegments.firstMile}
				bind:modes={preTransitModes}
				bind:maxTransitTime={maxPreTransitTime}
				possibleModes={prePostDirectModes}
				possibleMaxTransitTime={possiblePrePostDurations}
				ignoreRentalReturnConstraints={ignorePreTransitRentalReturnConstraints}
			></StreetModes>

			<!-- Last mile -->
			<StreetModes
				label={t.routingSegments.lastMile}
				bind:modes={postTransitModes}
				bind:maxTransitTime={maxPostTransitTime}
				possibleModes={prePostDirectModes}
				possibleMaxTransitTime={possiblePrePostDurations}
				ignoreRentalReturnConstraints={ignorePostTransitRentalReturnConstraints}
			></StreetModes>

			<!-- Direct -->
			{#if directModes !== undefined && maxDirectTime !== undefined && ignoreDirectRentalReturnConstraints !== undefined}
				<StreetModes
					label={t.routingSegments.direct}
					bind:modes={directModes}
					bind:maxTransitTime={maxDirectTime}
					possibleModes={prePostDirectModes}
					possibleMaxTransitTime={possibleDirectDurations}
					ignoreRentalReturnConstraints={ignoreDirectRentalReturnConstraints}
				></StreetModes>
			{/if}
		</div>

		<!-- Elevation Costs -->
		<div class="grid grid-cols-2 items-center">
			<div class="text-sm">
				{t.selectElevationCosts}
			</div>
			<Select.Root disabled={!allowElevationCosts} type="single" bind:value={elevationCosts}>
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
	</div>
{/if}
