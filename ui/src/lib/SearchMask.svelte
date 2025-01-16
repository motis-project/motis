<script lang="ts">
	import ArrowUpDown from 'lucide-svelte/icons/arrow-up-down';
	import Accessibility from 'lucide-svelte/icons/accessibility';
	import Bike from 'lucide-svelte/icons/bike';
	import AddressTypeahead from '$lib/AddressTypeahead.svelte';
	import Button from '$lib/components/ui/button/button.svelte';
	import { Label } from '$lib/components/ui/label';
	import * as RadioGroup from '$lib/components/ui/radio-group';
	import DateInput from '$lib/DateInput.svelte';
	import { type Location } from '$lib/Location';
	import { Toggle } from '$lib/components/ui/toggle';
	import { t } from '$lib/i18n/translation';

	let {
		from = $bindable(),
		to = $bindable(),
		time = $bindable(),
		timeType = $bindable(),
		wheelchair = $bindable(),
		bikeRental = $bindable()
	}: {
		from: Location;
		to: Location;
		time: Date;
		timeType: string;
		wheelchair: boolean;
		bikeRental: boolean;
	} = $props();

	let fromItems = $state<Array<Location>>([]);
	let toItems = $state<Array<Location>>([]);
</script>

<div class="flex flex-col space-y-4 p-4">
	<AddressTypeahead name="from" placeholder={t.from} bind:selected={from} bind:items={fromItems} />
	<AddressTypeahead name="to" placeholder={t.to} bind:selected={to} bind:items={toItems} />
	<Button
		class="absolute z-10 right-12 top-10"
		variant="outline"
		size="icon"
		onclick={() => {
			const tmp = to;
			to = from;
			from = tmp;

			const tmpItems = toItems;
			toItems = fromItems;
			fromItems = tmpItems;
		}}
	>
		<ArrowUpDown class="w-5 h-5" />
	</Button>
	<div class="flex flex-row gap-2 flex-wrap">
		<DateInput bind:value={time} />
		<RadioGroup.Root class="flex" bind:value={timeType}>
			<Label
				for="departure"
				class="flex items-center rounded-md border-2 border-muted bg-popover p-1 px-2 hover:bg-accent hover:text-accent-foreground [&:has([data-state=checked])]:border-blue-600 hover:cursor-pointer"
			>
				<RadioGroup.Item value="departure" id="departure" class="sr-only" aria-label="Abfahrt" />
				<span>{t.departure}</span>
			</Label>
			<Label
				for="arrival"
				class="flex items-center rounded-md border-2 border-muted bg-popover p-1 px-2 hover:bg-accent hover:text-accent-foreground [&:has([data-state=checked])]:border-blue-600 hover:cursor-pointer"
			>
				<RadioGroup.Item value="arrival" id="arrival" class="sr-only" aria-label="Ankunft" />
				<span>{t.arrival}</span>
			</Label>
		</RadioGroup.Root>
		<div>
			<Toggle aria-label="toggle bold" bind:pressed={wheelchair}>
				<Accessibility class="h-6 w-6" />
			</Toggle>
			<Toggle aria-label="toggle bold" bind:pressed={bikeRental}>
				<Bike class="h-6 w-6" />
			</Toggle>
		</div>
	</div>
</div>
