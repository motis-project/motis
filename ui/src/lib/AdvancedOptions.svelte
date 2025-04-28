<script lang="ts">
	import Button from '$lib/components/ui/button/button.svelte';
	import { t } from '$lib/i18n/translation';
	import { Select, Tooltip } from 'bits-ui';
	import BusFront from 'lucide-svelte/icons/bus-front';
	import ChartNoAxesCombined from 'lucide-svelte/icons/chart-no-axes-combined';
	import ChevronsUpDown from 'lucide-svelte/icons/chevrons-up-down';
	import ChevronUp from 'lucide-svelte/icons/chevron-up';
	import ChevronsUp from 'lucide-svelte/icons/chevrons-up';
	import ChevronDown from 'lucide-svelte/icons/chevron-down';
	import ChevronsDown from 'lucide-svelte/icons/chevrons-down';
	import Check from 'lucide-svelte/icons/check';
	import { Switch } from './components/ui/switch';
	import type { ElevationCosts } from './openapi';

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
	const possibleElevationCosts = [
		{ value: 'NONE' as ElevationCosts, label: t.elevationCostsNone },
		{ value: 'LOW' as ElevationCosts, label: t.elevationCostsLow },
		{ value: 'HIGH' as ElevationCosts, label: t.elevationCostsHigh }
	];
	// eslint-disable-next-line  @typescript-eslint/no-explicit-any
	const modes = possibleModes.map((m) => ({ value: m, label: (t as any)[m] }));

	const selectedModeLabel = $derived(
		selectedModes.length && selectedModes.length != possibleModes.length
			? modes
					.filter((m) => selectedModes.includes(m.value))
					.map((m) => m.label)
					.join(', ')
			: t.defaultSelectedModes
	);
	// Reset value, if no valid value is passed, e.g. via query parameter
	elevationCosts =
		possibleElevationCosts.find((c) => c.value == elevationCosts)?.value ??
		possibleElevationCosts[0].value;
	const elevationCostsLabel = $derived(
		possibleElevationCosts.find((c) => c.value == elevationCosts)?.label ??
			possibleElevationCosts[0].label
	);

	let expanded = $state<boolean>(false);
	let elevationProfileAllowed = $derived(
		// requires any street mode supporting elevation profiles
		bikeCarriage
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
	<div class="w-full space-y-2">
		<Select.Root
			type="multiple"
			bind:value={selectedModes}
			onOpenChange={(o: boolean) => {
				if (o && !selectedModes.length) selectedModes = [...possibleModes];
			}}
		>
			<Select.Trigger
				class="flex items-center h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm transition-colors file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
				aria-label={t.selectModes}
			>
				<BusFront class="mr-[9px] size-6 text-muted-foreground shrink-0" />
				<div class="grow text-ellipsis overflow-hidden text-nowrap">{selectedModeLabel}</div>
				<ChevronsUpDown class="ml-auto size-6 text-muted-foreground shrink-0" />
			</Select.Trigger>
			<Select.Portal>
				<Select.Content
					class="z-10 max-h-96 w-[var(--bits-select-anchor-width)] min-w-[var(--bits-select-anchor-width)] rounded-xl border border-muted bg-background px-1 py-3 shadow-popover outline-none"
					sideOffset={10}
				>
					<Select.ScrollUpButton class="flex w-full items-center justify-center">
						<ChevronsUp class="size-3" />
					</Select.ScrollUpButton>
					<Select.Viewport class="p-1">
						{#each modes as mode, i (i + mode.value)}
							<Select.Item
								class="flex h-10 w-full select-none items-center rounded-button py-3 pl-5 pr-1.5 text-sm outline-none duration-75 data-[highlighted]:bg-muted"
								value={mode.value}
								label={mode.label}
							>
								{#snippet children({ selected }: { selected: boolean })}
									{mode.label}
									{#if selected}
										<div class="ml-auto">
											<Check />
										</div>
									{/if}
								{/snippet}
							</Select.Item>
						{/each}
					</Select.Viewport>
					<Select.ScrollDownButton class="flex w-full items-center justify-center">
						<ChevronsDown class="size-3" />
					</Select.ScrollDownButton>
				</Select.Content>
			</Select.Portal>
		</Select.Root>

		<Switch bind:checked={wheelchair} label={t.wheelchair} id="wheelchair" />
		<Switch bind:checked={bikeRental} label={t.bikeRental} id="bikeRental" />
		<Switch bind:checked={bikeCarriage} label={t.bikeCarriage} id="bikeCarriage" />

		{#if elevationProfileAllowed}
			<div class="ml-4 space-y-2">
				<Tooltip.Provider>
					<Tooltip.Root delayDuration={150}>
						<Tooltip.Trigger
							class="w-full border-border-input bg-background-alt hover:bg-muted focus-visible:outline-hidden inline-flex"
						>
							<Select.Root type="single" bind:value={elevationCosts} items={possibleElevationCosts}>
								<Select.Trigger
									class="flex items-center h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm transition-colors file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
									aria-label={t.selectElevationCosts}
								>
									<ChartNoAxesCombined class="mr-[9px] size-6 text-muted-foreground shrink-0" />
									<div class="grow text-ellipsis overflow-hidden text-nowrap">
										{elevationCostsLabel}
									</div>
									<ChevronsUpDown class="ml-auto size-6 text-muted-foreground shrink-0" />
								</Select.Trigger>
								<Select.Portal>
									<Select.Content
										class="z-10 max-h-96 w-[var(--bits-select-anchor-width)] min-w-[var(--bits-select-anchor-width)] rounded-xl border border-muted bg-background px-1 py-3 shadow-popover outline-none"
										sideOffset={10}
									>
										<Select.ScrollUpButton class="flex w-full items-center justify-center">
											<ChevronsUp class="size-3" />
										</Select.ScrollUpButton>
										<Select.Viewport class="p-1">
											{#each possibleElevationCosts as costs, i (i + costs.value)}
												<Select.Item
													class="flex h-10 w-full select-none items-center rounded-button py-3 pl-5 pr-1.5 text-sm outline-none duration-75 data-[highlighted]:bg-muted"
													value={costs.value}
													label={costs.label}
												>
													{#snippet children({ selected }: { selected: boolean })}
														{costs.label}
														{#if selected}
															<div class="ml-auto">
																<Check />
															</div>
														{/if}
													{/snippet}
												</Select.Item>
											{/each}
										</Select.Viewport>
										<Select.ScrollDownButton class="flex w-full items-center justify-center">
											<ChevronsDown class="size-3" />
										</Select.ScrollDownButton>
									</Select.Content>
								</Select.Portal>
							</Select.Root>
						</Tooltip.Trigger>
						<Tooltip.Content sideOffset={8}>
							<div
								class="rounded-input border-dark-10 bg-background shadow-popover outline-hidden z-0 flex items-center justify-center border p-3 text-sm font-medium"
							>
								{t.selectElevationCosts}
							</div>
						</Tooltip.Content>
					</Tooltip.Root>
				</Tooltip.Provider>
			</div>
		{/if}
		<div class="text-muted-foreground leading-tight">{t.unreliableOptions}</div>
	</div>
{/if}
