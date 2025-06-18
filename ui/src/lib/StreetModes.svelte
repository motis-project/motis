<script lang="ts">
	import { t } from '$lib/i18n/translation';
	import { cn } from '$lib/utils';
	import * as Select from '$lib/components/ui/select';
	import { Switch } from '$lib/components/ui/switch';
	import { formatDurationSec } from '$lib/formatDuration';
	import { type PrePostDirectMode } from '$lib/Modes';

	let {
		label,
		modes = $bindable(),
		maxTransitTime = $bindable(),
		possibleModes,
		possibleMaxTransitTime,
		ignoreRentalReturnConstraints = $bindable()
	}: {
		label: string;
		modes: PrePostDirectMode[];
		maxTransitTime: number;
		possibleModes: readonly PrePostDirectMode[];
		possibleMaxTransitTime: number[];
		ignoreRentalReturnConstraints: boolean;
	} = $props();

	type TranslationKey = keyof typeof t;

	const availableModes = possibleModes.map((value) => ({
		value,
		label: t[value as TranslationKey] as string
	}));

	const containsRental = (modes: PrePostDirectMode[]) =>
		modes.some((mode) => mode.startsWith('RENTAL_'));

	const showRental = $derived(containsRental(modes));

	const selectedModesLabel = $derived(
		availableModes
			.filter((m) => modes?.includes(m.value))
			.map((m) => m.label)
			.join(', ')
	);
</script>

<div class="grid grid-cols-[1fr_2fr_1fr] items-center gap-2">
	<div class="text-sm">
		{label}
	</div>
	<Select.Root type="multiple" bind:value={modes}>
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
	<div class={cn('col-span-2 col-start-2', showRental || 'hidden')}>
		<Switch
			bind:checked={
				() => !ignoreRentalReturnConstraints, (v) => (ignoreRentalReturnConstraints = !v)
			}
			label={t.considerRentalReturnConstraints}
			id="ignorePreTransitRentalReturnConstraints"
		/>
	</div>
</div>
