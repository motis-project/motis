<script lang="ts">
	import ArrowUpDown from 'lucide-svelte/icons/arrow-up-down';
	import LocateFixed from 'lucide-svelte/icons/locate-fixed';
	import AddressTypeahead from '$lib/AddressTypeahead.svelte';
	import Button from '$lib/components/ui/button/button.svelte';
	import { Label } from '$lib/components/ui/label';
	import * as RadioGroup from '$lib/components/ui/radio-group';
	import DateInput from '$lib/DateInput.svelte';
	import { posToLocation, type Location } from '$lib/Location';
	import { t } from '$lib/i18n/translation';
	import AdvancedOptions from './AdvancedOptions.svelte';
	import maplibregl from 'maplibre-gl';

	let {
		geocodingBiasPlace,
		from = $bindable(),
		to = $bindable(),
		time = $bindable(),
		timeType = $bindable(),
		wheelchair = $bindable(),
		bikeRental = $bindable(),
		bikeCarriage = $bindable(),
		selectedModes = $bindable()
	}: {
		geocodingBiasPlace?: maplibregl.LngLatLike;
		from: Location;
		to: Location;
		time: Date;
		timeType: string;
		wheelchair: boolean;
		bikeRental: boolean;
		bikeCarriage: boolean;
		selectedModes: string[];
	} = $props();

	let fromItems = $state<Array<Location>>([]);
	let toItems = $state<Array<Location>>([]);

	const getLocation = () => {
		if (navigator && navigator.geolocation) {
			navigator.geolocation.getCurrentPosition(applyPosition, (e) => console.log(e), {
				enableHighAccuracy: true
			});
		}
	};

	const applyPosition = (position: { coords: { latitude: number; longitude: number } }) => {
		from = posToLocation({ lat: position.coords.latitude, lon: position.coords.longitude }, 0);
	};
</script>

<div id="searchmask-container" class="flex flex-col space-y-4 p-4 relative">
	<AddressTypeahead
		place={geocodingBiasPlace}
		name="from"
		placeholder={t.from}
		bind:selected={from}
		bind:items={fromItems}
	/>
	<AddressTypeahead
		place={geocodingBiasPlace}
		name="to"
		placeholder={t.to}
		bind:selected={to}
		bind:items={toItems}
	/>
	<Button
		variant="ghost"
		class="absolute z-10 right-4 top-0"
		size="icon"
		onclick={() => getLocation()}
	>
		<LocateFixed class="w-5 h-5" />
	</Button>
	<Button
		class="absolute z-10 right-14 top-6"
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
				<RadioGroup.Item
					value="departure"
					id="departure"
					class="sr-only"
					aria-label={t.departure}
				/>
				<span>{t.departure}</span>
			</Label>
			<Label
				for="arrival"
				class="flex items-center rounded-md border-2 border-muted bg-popover p-1 px-2 hover:bg-accent hover:text-accent-foreground [&:has([data-state=checked])]:border-blue-600 hover:cursor-pointer"
			>
				<RadioGroup.Item value="arrival" id="arrival" class="sr-only" aria-label={t.arrival} />
				<span>{t.arrival}</span>
			</Label>
		</RadioGroup.Root>
		<AdvancedOptions bind:wheelchair bind:bikeRental bind:bikeCarriage bind:selectedModes />
	</div>
</div>
