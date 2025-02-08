<script lang="ts">
	import Button from '$lib/components/ui/button/button.svelte';
	import { t } from '$lib/i18n/translation';
	import { Select } from 'bits-ui';
	import BusFront from 'lucide-svelte/icons/bus-front';
	import ChevronsUpDown from 'lucide-svelte/icons/chevrons-up-down';
	import ChevronUp from 'lucide-svelte/icons/chevron-up';
	import ChevronsUp from 'lucide-svelte/icons/chevrons-up';
	import ChevronDown from 'lucide-svelte/icons/chevron-down';
	import ChevronsDown from 'lucide-svelte/icons/chevrons-down';
	import Check from 'lucide-svelte/icons/check';
	import { Switch } from './components/ui/switch';

	let {
		selectedModes = $bindable(),
		wheelchair = $bindable(),
		bikeRental = $bindable(),
		bikeCarriage = $bindable()
	}: {
		selectedModes: string[];
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

	let expanded = $state<boolean>(false);
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
	<div class="w-full">
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
		<div class="text-muted-foreground leading-tight">{t.unreliableOptions}</div>
	</div>
{/if}
