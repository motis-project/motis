<script lang="ts">
	import Button from '$lib/components/ui/button/button.svelte';
	import { t } from '$lib/i18n/translation';
	import * as Select from '$lib/components/ui/select';
	import BusFront from 'lucide-svelte/icons/bus-front';
	import ChevronUp from 'lucide-svelte/icons/chevron-up';
	import ChevronDown from 'lucide-svelte/icons/chevron-down';
	import { Switch } from './components/ui/switch';
	import type { ElevationCosts } from '$lib/api/openapi';
	import { formatDurationSec } from './formatDuration';
	import { cn } from './utils';
	import {
		possibleTransitModes,
		prePostDirectModes,
		type PrePostDirectMode,
		type TransitMode
	} from './Modes';
	import StreetModes from './components/ui/street_modes.svelte';

	let {
		useRoutedTransfers = $bindable(),
		wheelchair = $bindable(),
		requireBikeTransport = $bindable(),
		requireCarTransport = $bindable(),
		transitModes = $bindable(),
		preTransitModes = $bindable(),
		postTransitModes = $bindable(),
		directModes = $bindable(),
		maxPreTransitTime = $bindable(),
		maxPostTransitTime = $bindable(),
		maxDirectTime = $bindable(),
		elevationCosts = $bindable(),
		ignorePreTransitRentalReturnConstraints = $bindable(),
		ignorePostTransitRentalReturnConstraints = $bindable(),
		ignoreDirectRentalReturnConstraints = $bindable()
	}: {
		useRoutedTransfers: boolean;
		wheelchair: boolean;
		requireBikeTransport: boolean;
		requireCarTransport: boolean;
		transitModes: TransitMode[];
		preTransitModes: PrePostDirectMode[];
		postTransitModes: PrePostDirectMode[];
		directModes: PrePostDirectMode[];
		maxPreTransitTime: number;
		maxPostTransitTime: number;
		maxDirectTime: number;
		elevationCosts: ElevationCosts;
		ignorePreTransitRentalReturnConstraints: boolean;
		ignorePostTransitRentalReturnConstraints: boolean;
		ignoreDirectRentalReturnConstraints: boolean;
	} = $props();

	type TranslationKey = keyof typeof t;

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
	const possiblePrePostDurations = [5 * 60, 10 * 60, 15 * 60, 20 * 60, 25 * 60, 30 * 60, 60 * 60];
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
	const availableTransitModes = possibleTransitModes.map((value) => ({
		value,
		label: t[value as TranslationKey] as string
	}));
	const availablePrePostDirectModes = prePostDirectModes.map((value) => ({
		value,
		label: t[value as TranslationKey] as string
	}));

	const selectTransitModesLabel = $derived(
		transitModes.length == possibleTransitModes.length || transitModes.includes('TRANSIT')
			? t.defaultSelectedModes
			: availableTransitModes
					.filter((m) => transitModes?.includes(m.value))
					.map((m) => m.label)
					.join(', ')
	);
	const selectedPreTransitModesLabel = $derived(
		availablePrePostDirectModes
			.filter((m) => preTransitModes?.includes(m.value))
			.map((m) => m.label)
			.join(', ')
	);
	const selectedPostTransitModesLabel = $derived(
		availablePrePostDirectModes
			.filter((m) => postTransitModes?.includes(m.value))
			.map((m) => m.label)
			.join(', ')
	);
	const selectedDirectModesLabel = $derived(
		availablePrePostDirectModes
			.filter((m) => directModes?.includes(m.value))
			.map((m) => m.label)
			.join(', ')
	);

	let expanded = $state<boolean>(false);
	let allowElevationCosts = $derived(
		requireBikeTransport ||
			preTransitModes.includes('BIKE') ||
			postTransitModes.includes('BIKE') ||
			directModes.includes('BIKE')
	);

	const containsRental = (modes: PrePostDirectMode[]) =>
		modes.some((mode) => mode.startsWith('RENTAL_'));
	const preTransitRental = $derived(containsRental(preTransitModes));
	const postTransitRental = $derived(containsRental(postTransitModes));
	const directRental = $derived(containsRental(directModes));
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
		<Select.Root type="multiple" bind:value={transitModes}>
			<Select.Trigger
				class="flex items-center w-full overflow-hidden"
				aria-label={t.selectTransitModes}
			>
				<BusFront class="mr-[9px] size-6 text-muted-foreground shrink-0" />
				{selectTransitModesLabel}
			</Select.Trigger>
			<Select.Content sideOffset={10}>
				{#each availableTransitModes as mode, i (i + mode.value)}
					<Select.Item value={mode.value} label={mode.label}>
						{mode.label}
					</Select.Item>
				{/each}
			</Select.Content>
		</Select.Root>

		<div class="space-y-2">
			<Switch
				bind:checked={useRoutedTransfers}
				label={t.useRoutedTransfers}
				id="useRoutedTransfers"
				onCheckedChange={() => {
					if (wheelchair && !useRoutedTransfers) {
						wheelchair = false;
					}
				}}
			/>
			<Switch
				bind:checked={wheelchair}
				label={t.wheelchair}
				id="wheelchair"
				onCheckedChange={() => {
					if (wheelchair && !useRoutedTransfers) {
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
				onCheckedChange={setModes('CAR')}
				id="requireCarTransport"
			/>
		</div>

		<div class="grid grid-cols-[1fr_2fr_1fr] items-center gap-2">
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
			<StreetModes
				label={t.routingSegments.direct}
				bind:modes={directModes}
				bind:maxTransitTime={maxDirectTime}
				possibleModes={prePostDirectModes}
				possibleMaxTransitTime={possibleDirectDurations}
				ignoreRentalReturnConstraints={ignoreDirectRentalReturnConstraints}
			></StreetModes>
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
	</div>
{/if}
