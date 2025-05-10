<script lang="ts">
	import Button from '$lib/components/ui/button/button.svelte';
	import { t } from '$lib/i18n/translation';
	import * as Select from '$lib/components/ui/select';
	import BusFront from 'lucide-svelte/icons/bus-front';
	import ChevronUp from 'lucide-svelte/icons/chevron-up';
	import ChevronDown from 'lucide-svelte/icons/chevron-down';
	import { Switch } from './components/ui/switch';
	import type { ElevationCosts, Mode } from './openapi';

	let {
		selectedModes = $bindable(),
		elevationCosts = $bindable(),
		wheelchair = $bindable(),
		bikeRental = $bindable(),
		bikeCarriage = $bindable(),
		carCarriage = $bindable(),
		firstMileMode = $bindable(),
		lastMileMode = $bindable(),
		directModes = $bindable()
	}: {
		selectedModes: string[] | undefined;
		elevationCosts: ElevationCosts;
		wheelchair: boolean;
		bikeRental: boolean;
		bikeCarriage: boolean;
		carCarriage: boolean;
		firstMileMode: Mode;
		lastMileMode: Mode;
		directModes: Mode[];
	} = $props();

	type TranslationKey = keyof typeof t;

	const possibleModes = [
		'AIRPLANE',
		'HIGHSPEED_RAIL',
		'LONG_DISTANCE',
		'NIGHT_RAIL',
		'COACH',
		'REGIONAL_FAST_RAIL',
		'REGIONAL_RAIL',
		'METRO',
		'SUBWAY',
		'TRAM',
		'BUS',
		'FERRY',
		'OTHER'
	];
	if (selectedModes === undefined) {
		selectedModes = [...possibleModes];
	}

	const possibleElevationCosts = [
		{ value: 'NONE' as ElevationCosts, label: t.elevationCosts.NONE },
		{ value: 'LOW' as ElevationCosts, label: t.elevationCosts.LOW },
		{ value: 'HIGH' as ElevationCosts, label: t.elevationCosts.HIGH }
	];
	const modes = possibleModes.map((m) => ({ value: m, label: t[m as TranslationKey] as string }));

	const selectedModeLabel = $derived(
		selectedModes.length != possibleModes.length
			? modes
					.filter((m) => selectedModes?.includes(m.value))
					.map((m) => m.label)
					.join(', ')
			: t.defaultSelectedModes
	);
	const selectedFirstMileModeLabel = $derived(
		t[firstMileMode as TranslationKey] +
			(bikeCarriage && firstMileMode != 'BIKE' ? `, ${t.bikeCarriage} (${t.BIKE})` : '') +
			(carCarriage && firstMileMode != 'CAR' ? `, ${t.carCarriage} (${t.CAR})` : '')
	);
	const selectedLastMileModeLabel = $derived(
		t[lastMileMode as TranslationKey] +
			(bikeCarriage && lastMileMode != 'BIKE' ? `, ${t.bikeCarriage} (${t.BIKE})` : '') +
			(carCarriage && lastMileMode != 'CAR' ? `, ${t.carCarriage} (${t.CAR})` : '')
	);
	const selectedDirectModesLabel = $derived(
		directModes.length || bikeCarriage || carCarriage
			? [
					...directModes.map((m) => t[m as TranslationKey]),
					...(bikeCarriage && !directModes.includes('BIKE')
						? [`${t.bikeCarriage} (${t.BIKE})`]
						: []),
					...(carCarriage && !directModes.includes('CAR') ? [`${t.carCarriage} (${t.CAR})`] : [])
				].join(', ')
			: `${t.default} (${t.WALK})`
	);

	let expanded = $state<boolean>(false);
	let allowElevationCosts = $derived(
		bikeCarriage ||
			firstMileMode == 'BIKE' ||
			lastMileMode == 'BIKE' ||
			directModes.includes('BIKE')
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
		<Select.Root type="multiple" bind:value={selectedModes}>
			<Select.Trigger class="flex items-center w-full overflow-hidden" aria-label={t.selectModes}>
				<BusFront class="mr-[9px] size-6 text-muted-foreground shrink-0" />
				{selectedModeLabel}
			</Select.Trigger>
			<Select.Content sideOffset={10}>
				{#each modes as mode, i (i + mode.value)}
					<Select.Item value={mode.value} label={mode.label}>
						{mode.label}
					</Select.Item>
				{/each}
			</Select.Content>
		</Select.Root>

		<div class="space-y-2">
			<Switch bind:checked={wheelchair} label={t.wheelchair} id="wheelchair" />
			<Switch bind:checked={bikeRental} label={t.bikeRental} id="bikeRental" />
			<Switch bind:checked={bikeCarriage} label={t.bikeCarriage} id="bikeCarriage" />
			<Switch bind:checked={carCarriage} label={t.carCarriage} id="carCarriage" />
		</div>

		<div class="grid grid-cols-[1fr_2fr] items-center space-y-2">
			<!-- First mile -->
			<div class="text-sm">
				{t.routingSegments.firstMile}
			</div>
			<Select.Root type="single" bind:value={firstMileMode}>
				<Select.Trigger
					class="flex items-center w-full overflow-hidden"
					aria-label={t.routingSegments.firstMile}
				>
					{selectedFirstMileModeLabel}
				</Select.Trigger>
				<Select.Content sideOffset={10}>
					{#each ['WALK', 'BIKE', 'CAR'] as mode, i (i + mode)}
						<Select.Item value={mode} label={t[mode as TranslationKey] as string}>
							{t[mode as TranslationKey]}
						</Select.Item>
					{/each}
				</Select.Content>
			</Select.Root>
			<!-- Last mile -->
			<div class="text-sm">
				{t.routingSegments.lastMile}
			</div>
			<Select.Root type="single" bind:value={lastMileMode}>
				<Select.Trigger
					class="flex items-center w-full overflow-hidden"
					aria-label={t.routingSegments.lastMile}
				>
					{selectedLastMileModeLabel}
				</Select.Trigger>
				<Select.Content sideOffset={10}>
					{#each ['WALK', 'BIKE', 'CAR'] as mode, i (i + mode)}
						<Select.Item value={mode} label={t[mode as TranslationKey] as string}>
							{t[mode as TranslationKey]}
						</Select.Item>
					{/each}
				</Select.Content>
			</Select.Root>
			<!-- Direct -->
			<div class="text-sm">
				{t.routingSegments.direct}
			</div>
			<Select.Root type="multiple" bind:value={directModes}>
				<Select.Trigger
					class="flex items-center w-full overflow-hidden"
					aria-label={t.routingSegments.direct}
				>
					{selectedDirectModesLabel}
				</Select.Trigger>
				<Select.Content sideOffset={10}>
					{#each ['WALK', 'BIKE', 'CAR'] as mode, i (i + mode)}
						<Select.Item value={mode} label={t[mode as TranslationKey] as string}>
							{t[mode as TranslationKey]}
						</Select.Item>
					{/each}
				</Select.Content>
			</Select.Root>
		</div>
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

		<div class="text-muted-foreground leading-tight">{t.unreliableOptions}</div>
	</div>
{/if}
