<script lang="ts">
	import { t } from '$lib/i18n/translation';
	import { BusFront } from '@lucide/svelte';
	import * as Select from '$lib/components/ui/select';
	import { possibleTransitModes, type TransitMode } from '$lib/Modes';

	let {
		transitModes = $bindable()
	}: {
		transitModes: TransitMode[];
	} = $props();

	type TranslationKey = keyof typeof t;

	const availableTransitModes = possibleTransitModes.map((value) => ({
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
</script>

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
