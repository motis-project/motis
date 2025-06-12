<script lang="ts">
	import * as Select from '$lib/components/ui/select';

	export type NumberSelectOption = { value: string; label: string };

	let {
		value = $bindable(),
		possibleValues,
		labelFormatter = (v) => v.toString()
	}: {
		value: number;
		possibleValues: NumberSelectOption[];
		labelFormatter?: (v: number) => string;
	} = $props();
</script>

<Select.Root
	type="single"
	bind:value={() => value.toString(), (v) => (value = parseInt(v))}
	items={possibleValues}
>
	<Select.Trigger class="flex items-center w-full overflow-hidden" aria-label="max travel time">
		<div class="w-full text-right pr-4">{labelFormatter(value)}</div>
	</Select.Trigger>
	<Select.Content align="end">
		{#each possibleValues as option, i (i + option.value)}
			<Select.Item value={option.value} label={option.label}>
				<div class="w-full text-right pr-2">{option.label}</div>
			</Select.Item>
		{/each}
	</Select.Content>
</Select.Root>
