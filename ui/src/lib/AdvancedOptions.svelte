<script lang="ts">
	import Button from '$lib/components/ui/button/button.svelte';
	import { t } from '$lib/i18n/translation';
	import * as Select from '$lib/components/ui/select';
	import BusFront from 'lucide-svelte/icons/bus-front';
	import ChevronUp from 'lucide-svelte/icons/chevron-up';
	import ChevronDown from 'lucide-svelte/icons/chevron-down';
	import * as RadioGroup from '$lib/components/ui/radio-group';
	import { Switch } from './components/ui/switch';
	import type { ElevationCosts } from './openapi';
	import { Label } from './components/ui/label';
	import { SvelteMap } from 'svelte/reactivity';

	let {
		selectedModes = $bindable(),
		elevationCosts = $bindable(),
		wheelchair = $bindable(),
		bikeRental = $bindable(),
		bikeCarriage = $bindable()
	}: {
		selectedModes: string[];
		elevationCosts: ElevationCosts;
		wheelchair: boolean;
		bikeRental: boolean;
		bikeCarriage: boolean;
	} = $props();

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
	const segments = ['firstMile', 'lastMile', 'direct'];
	type Segment = (typeof segments)[number];
	const streetModes = ['WALK', 'BIKE', 'CAR'];
	type StreetMode = (typeof streetModes)[number];

	let streetModeMap = new SvelteMap<Segment, StreetMode>([]);
	const getStreetMode = (segment: Segment) => streetModeMap.get(segment) ?? 'WALK';
	const streetModeFilter = (seg: Segment, mode: StreetMode) => {
		switch (seg) {
			case 'lastMile':
				return mode != 'CAR';
			default:
				return true;
		}
	};

	const possibleElevationCosts = [
		{ value: 'NONE' as ElevationCosts, label: t.elevationCosts.NONE },
		{ value: 'LOW' as ElevationCosts, label: t.elevationCosts.LOW },
		{ value: 'HIGH' as ElevationCosts, label: t.elevationCosts.HIGH }
	];
	// eslint-disable-next-line  @typescript-eslint/no-explicit-any
	const modes = possibleModes.map((m) => ({ value: m, label: (t as any)[m] }));

	const selectedModeLabel = $derived(
		selectedModes.length != possibleModes.length
			? modes
					.filter((m) => selectedModes.includes(m.value))
					.map((m) => m.label)
					.join(', ')
			: t.defaultSelectedModes
	);

	let expanded = $state<boolean>(false);
	let allowElevationCosts = $derived(
		bikeCarriage || streetModeMap.values().some((v) => v == 'BIKE')
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
		</div>

		<div class="flex flex-row w-full items-center space-x-2">
			{#each segments as segment}
				<div class="flex flex-col w-full">
					<div class="flex justify-center">{(t.routingSegments as any)[segment]}</div>
					<Select.Root
						type="single"
						bind:value={
							(): StreetMode => getStreetMode(segment),
							(v: StreetMode) => streetModeMap.set(segment, v)
						}
					>
						<Select.Trigger aria-label="Select modes for first mile">
							{(t as any)[getStreetMode(segment)]}
						</Select.Trigger>
						<Select.Content sideOffset={10}>
							{#each streetModes as mode, i (i + mode)}
								{#if streetModeFilter(segment, mode)}
									<Select.Item value={mode} label={(t as any)[mode]}>
										{(t as any)[mode]}
									</Select.Item>
								{/if}
							{/each}
						</Select.Content>
					</Select.Root>
				</div>
			{/each}
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
