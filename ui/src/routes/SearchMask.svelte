<script lang="ts">
	import AddressTypeahead from '$lib/AddressTypeahead.svelte';
	import Button from '$lib/components/ui/button/button.svelte';
	import { Label } from '$lib/components/ui/label';
	import * as RadioGroup from '$lib/components/ui/radio-group';
	import * as Select from '$lib/components/ui/select';
	import DateInput from '$lib/DateInput.svelte';
	import { type Location } from '$lib/Location';
	import type { Selected } from 'bits-ui';
	import ArrowUpDown from 'lucide-svelte/icons/arrow-up-down';

	let {
		from = $bindable(),
		to = $bindable(),
		dateTime = $bindable(),
		timeType = $bindable(),
		profile = $bindable(),
		theme
	}: {
		from: Location;
		to: Location;
		dateTime: Date;
		timeType: string;
		profile: Selected<string>;
		theme?: 'light' | 'dark';
	} = $props();

	let fromItems = $state<Array<Location>>([]);
	let toItems = $state<Array<Location>>([]);
</script>

<div class="flex flex-col space-y-4 p-4">
	<AddressTypeahead
		name="from"
		class="w-full"
		placeholder="From"
		bind:selected={from}
		bind:items={fromItems}
		{theme}
	/>
	<AddressTypeahead
		name="to"
		class="w-full"
		placeholder="To"
		bind:selected={to}
		bind:items={toItems}
		{theme}
	/>
	<Button
		class="absolute z-10 right-12 top-10"
		variant="outline"
		size="icon"
		on:click={() => {
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
	<div class="flex flex-row space-x-2 justify-between">
		<DateInput bind:value={dateTime} />
		<RadioGroup.Root class="flex space-x-1" bind:value={timeType}>
			<Label
				for="departure"
				class="flex items-center rounded-md border-2 border-muted bg-popover p-1 px-2 hover:bg-accent hover:text-accent-foreground [&:has([data-state=checked])]:border-blue-600 hover:cursor-pointer"
			>
				<RadioGroup.Item value="departure" id="departure" class="sr-only" aria-label="Abfahrt" />
				<span>Abfahrt</span>
			</Label>
			<Label
				for="arrival"
				class="flex items-center rounded-md border-2 border-muted bg-popover p-1 px-2 hover:bg-accent hover:text-accent-foreground [&:has([data-state=checked])]:border-blue-600 hover:cursor-pointer"
			>
				<RadioGroup.Item value="arrival" id="arrival" class="sr-only" aria-label="Ankunft" />
				<span>Ankunft</span>
			</Label>
		</RadioGroup.Root>
		<div class="min-w-22">
			<Select.Root bind:selected={profile}>
				<Select.SelectTrigger>
					<Select.SelectValue placeholder="Profile" />
				</Select.SelectTrigger>
				<Select.SelectContent>
					<Select.SelectItem value="wheelchair">Wheelchair</Select.SelectItem>
					<Select.SelectItem value="foot">Foot</Select.SelectItem>
					<Select.SelectItem value="bike">Bike</Select.SelectItem>
					<Select.SelectItem value="car">Car</Select.SelectItem>
				</Select.SelectContent>
			</Select.Root>
		</div>
	</div>
</div>
