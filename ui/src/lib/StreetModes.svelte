<script lang="ts">
	import { t } from '$lib/i18n/translation';
	import { cn } from '$lib/utils';
	import * as Select from '$lib/components/ui/select';
	import { Switch } from '$lib/components/ui/switch';
	import { formatDurationSec } from '$lib/formatDuration';
	import { type PrePostDirectMode, getFormFactors } from '$lib/Modes';
	import { formFactorAssets } from '$lib/map/rentals/assets';
	import { DEFAULT_COLOR } from '$lib/map/rentals/style';
	import { rentals, type RentalFormFactor } from '@motis-project/motis-client';
	import { createQuery } from '@tanstack/svelte-query';

	let {
		label,
		disabled,
		modes = $bindable(),
		maxTransitTime = $bindable(),
		possibleModes,
		possibleMaxTransitTime,
		ignoreRentalReturnConstraints = $bindable(),
		providerGroups = $bindable()
	}: {
		label: string;
		disabled?: boolean;
		modes: PrePostDirectMode[];
		maxTransitTime: number;
		possibleModes: readonly PrePostDirectMode[];
		possibleMaxTransitTime: number[];
		ignoreRentalReturnConstraints: boolean;
		providerGroups: string[];
	} = $props();

	type TranslationKey = keyof typeof t;

	const availableModes = possibleModes.map((value) => ({
		value,
		label: t[value as TranslationKey] as string
	}));

	type ProviderOption = {
		id: string;
		name: string;
		color: string;
		formFactors: RentalFormFactor[];
	};

	const providerGroupsQuery = createQuery(() => ({
		queryKey: ['rentalProviderGroups'],
		queryFn: async () => {
			const { data, error } = await rentals({ query: { withProviders: false } });
			if (error) {
				throw error;
			}
			return data;
		},
		staleTime: 0
	}));

	const allProviderGroupOptions = $derived.by((): ProviderOption[] => {
		return (providerGroupsQuery.data?.providerGroups || [])
			.map((group) => ({
				id: group.id,
				name: group.name,
				color: group.color ?? DEFAULT_COLOR,
				formFactors: group.formFactors
			}))
			.sort((a, b) => a.name.localeCompare(b.name));
	});

	const selectedRentalFormFactors = $derived(getFormFactors(modes));

	const providerGroupOptions = $derived.by((): ProviderOption[] => {
		if (selectedRentalFormFactors.length === 0) {
			return allProviderGroupOptions;
		}
		return allProviderGroupOptions.filter((option) => {
			return option.formFactors.some((formFactor) =>
				selectedRentalFormFactors.includes(formFactor)
			);
		});
	});

	const containsRental = (modes: PrePostDirectMode[]) =>
		modes.some((mode) => mode.startsWith('RENTAL_'));

	const showRental = $derived(containsRental(modes));

	$effect(() => {
		if (!providerGroupOptions.length) {
			return;
		}
		const deduped = Array.from(new Set(providerGroups));
		if (deduped.length !== providerGroups.length) {
			providerGroups = deduped;
			return;
		}
		const validIds = new Set(providerGroupOptions.map((option) => option.id));
		const filtered = deduped.filter((id) => validIds.has(id));
		if (filtered.length !== deduped.length) {
			providerGroups = filtered;
			return;
		}
		if (filtered.length && filtered.length === validIds.size) {
			providerGroups = [];
		}
	});

	const selectedModesLabel = $derived(
		availableModes
			.filter((m) => modes?.includes(m.value))
			.map((m) => m.label)
			.join(', ')
	);

	const selectedProviderGroupsLabel = $derived.by(() => {
		const names = providerGroupOptions
			.filter((option) => providerGroups.includes(option.id))
			.map((option) => option.name);
		return names.length ? names.join(', ') : t.defaultSelectedProviders;
	});
</script>

<div class="grid grid-cols-[1fr_2fr_1fr] items-center gap-2">
	<div class="text-sm">
		{label}
	</div>
	<Select.Root {disabled} type="multiple" bind:value={modes}>
		<Select.Trigger class="flex items-center w-full overflow-hidden" aria-label={label}>
			<span>{selectedModesLabel}</span>
		</Select.Trigger>
		<Select.Content sideOffset={10}>
			{#each possibleModes as mode, i (i + mode)}
				<Select.Item value={mode} label={t[mode as TranslationKey] as string}>
					{t[mode as TranslationKey]}
				</Select.Item>
			{/each}
		</Select.Content>
	</Select.Root>
	<Select.Root
		{disabled}
		type="single"
		bind:value={() => maxTransitTime.toString(), (v) => (maxTransitTime = parseInt(v))}
	>
		<Select.Trigger
			class="flex items-center w-full overflow-hidden"
			aria-label={t.routingSegments.maxPreTransitTime}
		>
			{formatDurationSec(maxTransitTime)}
		</Select.Trigger>
		<Select.Content sideOffset={10}>
			{#each possibleMaxTransitTime as duration (duration)}
				<Select.Item value={`${duration}`} label={formatDurationSec(duration)}>
					{formatDurationSec(duration)}
				</Select.Item>
			{/each}
		</Select.Content>
	</Select.Root>
	<div class={cn('text-sm', showRental || 'hidden')}>
		{t.sharingProviders}
	</div>
	<div class={cn('col-span-2 col-start-2', showRental || 'hidden')}>
		<Select.Root {disabled} type="multiple" bind:value={providerGroups}>
			<Select.Trigger
				class="flex items-center w-full overflow-hidden"
				aria-label={t.sharingProviders}
			>
				<span>{selectedProviderGroupsLabel}</span>
			</Select.Trigger>
			<Select.Content sideOffset={10} class="max-w-[100svw]">
				{#each providerGroupOptions as option (option.id)}
					<Select.Item value={option.id} label={option.name}>
						<div class="flex w-full items-center justify-between gap-2">
							<span class="truncate">{option.name}</span>
							{#if option.formFactors.length}
								<div class="flex items-center gap-1" style={`color: ${option.color}`}>
									{#each Array.from(new Set(option.formFactors.map((ff) => formFactorAssets[ff].svg))).sort() as icon (icon)}
										<svg class="h-4 w-4 fill-current" aria-hidden="true" focusable="false">
											<use href={`#${icon}`} />
										</svg>
									{/each}
								</div>
							{/if}
						</div>
					</Select.Item>
				{/each}
			</Select.Content>
		</Select.Root>
	</div>
	<div class={cn('col-span-2 col-start-2', showRental || 'hidden')}>
		<Switch
			{disabled}
			bind:checked={
				() => !ignoreRentalReturnConstraints, (v) => (ignoreRentalReturnConstraints = !v)
			}
			label={t.considerRentalReturnConstraints}
			id="ignorePreTransitRentalReturnConstraints"
		/>
	</div>
</div>
