<script lang="ts">
	import AddressTypeahead from '$lib/AddressTypeahead.svelte';
	import { type Location } from '$lib/Location';
	import { t } from '$lib/i18n/translation';
	import * as Select from '$lib/components/ui/select';
	import { Slider } from 'bits-ui';
	import { untrack } from 'svelte';
	import { oneToAll, plan, type OneToAllData, type OneToAllResponse, type ReachablePlace } from './api/openapi';
	import { lngLatToStr } from './lngLatToStr';

    interface IsochronesPos {
        lat: number;
        lng: number;
        duration: number;
        name?: string;
    };
	const toPlaceString = (l: Location) => {
		if (l.value.match?.type === 'STOP') {
			return l.value.match.id;
		} else if (l.value.match?.level) {
			return `${lngLatToStr(l.value.match!)},${l.value.match.level}`;
		} else {
			return `${lngLatToStr(l.value.match!)},0`;
		}
	};

	let {
		geocodingBiasPlace,
		isochronesData = $bindable(),
		time = $bindable()
	}: {
		geocodingBiasPlace?: maplibregl.LngLatLike;
		isochronesData: IsochronesPos[];
		time: Date;
	} = $props();

	const timeout = 60;

	const maxOption = 60;
	const optionSteps = 5;
	const possibleTravelTimes = [1, ...Array(Math.round(maxOption / optionSteps)).keys().map(i => (i + 1) * optionSteps)]
		.map(i => i.toString())
		.map(v => ( {value: v, label: v + " min"} ))
	;
	let from = $state<Location>() as Location;
	let fromItems = $state<Array<Location>>([]);
	let travelTime = $state("45");
	const maxTravelTime = $derived(parseInt(travelTime));

	let queryTimeout: number;

	let isochronesQuery = $derived(
		from?.value?.match
			? ({query: {
				one: toPlaceString(from),
				maxTravelTime: maxTravelTime
			}}) as OneToAllData
			: undefined
	);
	$effect(() => {
		if (isochronesQuery) {
			console.log("NEW QUERY", maxTravelTime);
			clearTimeout(queryTimeout);
			queryTimeout = setTimeout(() => {
				oneToAll(isochronesQuery)
					.then((r: {data: OneToAllResponse | undefined; error: unknown}) => {
						console.log("GOT RESPONSE");
						if (r.error) {
							throw new Error(String(r.error));
						}
						console.log("DATA OK");
						const all = r.data!.all!.map((p: ReachablePlace) => {
							return {
								lat: p.place?.lat,
								lng: p.place?.lon,
								duration: maxTravelTime - (p.duration ?? 0),
								name: p.place?.name,
							} as IsochronesPos
						})
						;
						console.log("ALL Calculated");
						console.log(all);
						untrack(() => {
							console.log("UPDATE 11");
							isochronesData = [...all];
							console.log("UPDATE 22");
						})
					})
				;
			}, timeout);
		}
	});
	// $effect(() => {
	// 	console.log('NEW LOCATION');
	// 	const lat = Math.random() + 50.5;
	// 	const lng = Math.random() + 6.8;
	// 	const dur = Math.random() * 15 + 5;
	// 	const name = `Test: ${maxTravelTime}`;
	// 	console.log("TO PUSH");
	// 	untrack(() => isochronesData.push({lat: lat, lng: lng, duration: dur, name: name}));
	// 	console.log("PUSHED");
	// });
</script>

<div id="searchmask-container" class="flex flex-col space-y-4 p-4 relative">
	<AddressTypeahead
		place={geocodingBiasPlace}
		name="from"
		placeholder={t.from}
		bind:selected={from}
		bind:items={fromItems}
	/>


	<div class="grid grid-cols-2 items-center">
		<div class="text-sm">
			Max travel time
		</div>
		<Select.Root type="single" bind:value={travelTime} items={possibleTravelTimes}>
			<Select.Trigger class="flex items-center w-full overflow-hidden" aria-label="max travel time">
				<div class="w-full text-right pr-4">{travelTime} min</div>
			</Select.Trigger>
			<Select.Content align="end">
				{#each possibleTravelTimes as option, i ( i + option.value )}
					<Select.Item value={option.value} label={option.label}>
						<div class="w-full text-right pr-2">{option.label}</div>
					</Select.Item>
				{/each}
			</Select.Content>
		</Select.Root>

		<div class="text-sm">
			Max travel time ({maxTravelTime})
		</div>
		<Slider.Root type="single" min={1} max={90} step={1} value={maxTravelTime} onValueChange={(v) => (travelTime = v.toString())} class="relative flex w-full touch-none select-none items-center">
			{#snippet  children()}
				<span class="bg-dark-10 relative h-2 w-full grow cursor-pointer overflow-hidden rounded-full">
					<Slider.Range class="bg-foreground absolute h-full" />
				</span>
				<Slider.Thumb
					index={0}
					class="border-border-input bg-background hover:border-dark-40 focus-visible:ring-foreground dark:bg-foreground dark:shadow-card focus-visible:outline-hidden block size-[25px] cursor-pointer rounded-full border shadow-sm transition-colors focus-visible:ring-2 focus-visible:ring-offset-2 active:scale-[0.98] disabled:pointer-events-none disabled:opacity-50"
					/>
			{/snippet}
		</Slider.Root>
	</div>
</div>
