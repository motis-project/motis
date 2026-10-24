<script lang="ts">
	import { RadioGroup } from 'bits-ui';
	import { Label } from '$lib/components/ui/label';
	import { t } from '$lib/i18n/translation';

	let {
		arriveBy = $bindable()
	}: {
		arriveBy: boolean;
	} = $props();

	const id = $props.id();
</script>

<RadioGroup.Root
	class="flex gap-1"
	bind:value={() => (arriveBy ? 'arrival' : 'departure'), (v) => (arriveBy = v === 'arrival')}
>
	<Label
		for="{id}-departure"
		class="flex items-center rounded-md border-2 border-muted bg-popover p-1 px-2 hover:bg-accent hover:text-accent-foreground [&:has([data-state=checked])]:border-blue-600 hover:cursor-pointer"
	>
		<RadioGroup.Item
			value="departure"
			id="{id}-departure"
			class="sr-only"
			aria-label={t.departure}
		/>
		<span>{t.departure}</span>
	</Label>
	<Label
		for="{id}-arrival"
		class="flex items-center rounded-md border-2 border-muted bg-popover p-1 px-2 hover:bg-accent hover:text-accent-foreground [&:has([data-state=checked])]:border-blue-600 hover:cursor-pointer"
	>
		<RadioGroup.Item value="arrival" id="{id}-arrival" class="sr-only" aria-label={t.arrival} />
		<span>{t.arrival}</span>
	</Label>
</RadioGroup.Root>
