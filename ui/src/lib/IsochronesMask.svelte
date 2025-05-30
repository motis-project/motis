<script lang="ts">
	import AddressTypeahead from '$lib/AddressTypeahead.svelte';
	import Button from '$lib/components/ui/button/button.svelte';
	import { posToLocation, type Location } from '$lib/Location';
	import { t } from '$lib/i18n/translation';
	import * as Select from '$lib/components/ui/select';
	import { Label } from '$lib/components/ui/label';
	import * as RadioGroup from '$lib/components/ui/radio-group';
	import ChevronUp from 'lucide-svelte/icons/chevron-up';
	import ChevronDown from 'lucide-svelte/icons/chevron-down';
	import LocateFixed from 'lucide-svelte/icons/locate-fixed';
	import { Slider } from 'bits-ui';
	import { untrack } from 'svelte';
	import { oneToAll, plan, type OneToAllData, type OneToAllResponse, type ReachablePlace } from './api/openapi';
	import { lngLatToStr } from './lngLatToStr';
	import DateInput from './DateInput.svelte';

    interface IsochronesPos {
        lat: number;
        lng: number;
        duration: number;
        name?: string;
    };
	const toPlaceString = (l: Location) => {
		if (l.match?.type === 'STOP') {
			return l.match.id;
		} else if (l.match?.level) {
			return `${lngLatToStr(l.match!)},${l.match.level}`;
		} else {
			return `${lngLatToStr(l.match!)},0`;
		}
	};

	let {
		from,
		to,
		// maxTravelTime = $bindable(),
		geocodingBiasPlace,
		isochronesData = $bindable(),
		time = $bindable(),
		arriveBy = $bindable(),
		color = $bindable(),
		opacity = $bindable()
	}: {
		from: Location;
		to: Location;
		// maxTravelTime: string;
		geocodingBiasPlace?: maplibregl.LngLatLike;
		isochronesData: IsochronesPos[];
		time: Date;
		arriveBy: boolean;
		color: string;
		opacity: number;
	} = $props();

	type TranslationKey = keyof typeof t;

	const timeout = 60;

	const maxOption = 60;
	const optionSteps = 5;
	const possibleTravelTimes = [1, ...Array(Math.round(maxOption / optionSteps)).keys().map(i => (i + 1) * optionSteps)]
		.map(i => i.toString())
		.map(v => ( {value: v, label: v + " min"} ))
	;
	let expanded = $state<boolean>(false);
	// let from = $state<Location>() as Location;
	// let fromItems = $state<Array<Location>>([]);
	let one = $state<Location>(from);
	let maxTravelTime = $state("45");
	let oneMileMode = $state("WALK");
	let maxOneTime = $state("15");

	const selectedMaxTravelTime = $derived(parseInt(maxTravelTime));
	const selectedMaxOneTimeSeconds = $derived(parseInt(maxOneTime) * 60);

	let lastFrom: Location = from;
	let lastTo: Location = to;
	let queryTimeout: number;

	let isochronesQuery = $derived(
		one?.match
			? ({query: {
				one: toPlaceString(one),
				maxTravelTime: selectedMaxTravelTime,
				time: time.toISOString(),
				arriveBy,
				preTransitModes: arriveBy ? undefined : oneMileMode,
				postTransitModes: arriveBy ? oneMileMode : undefined,
				maxPreTransitTime: arriveBy ? undefined : selectedMaxOneTimeSeconds,
				maxPostTransitTime: arriveBy ? selectedMaxOneTimeSeconds : undefined,
			}}) as OneToAllData
			: undefined
	);
	$effect(() => {
		if (isochronesQuery) {
			console.log("NEW QUERY", selectedMaxTravelTime);
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
								duration: selectedMaxTravelTime - (p.duration ?? 0),
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

	$effect(() => {
		if (lastFrom != from) {
			one = from;
			lastFrom = from;
		}
		if (lastTo != to) {
			one = to;
			lastTo = to;
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

	const getLocation = () => {
		if (navigator && navigator.geolocation) {
			navigator.geolocation.getCurrentPosition(applyPosition, (e) => console.log(e), {
				enableHighAccuracy: true
			});
		}
	};

	const applyPosition = (position: { coords: { latitude: number; longitude: number } }) => {
		one = posToLocation({ lat: position.coords.latitude, lon: position.coords.longitude }, 0);
	};
</script>

<div id="searchmask-container" class="flex flex-col space-y-4 p-4 relative">
	<AddressTypeahead
		place={geocodingBiasPlace}
		name="one"
		placeholder={t.from}
		bind:selected={one}
	/>
	<Button
		variant="ghost"
		class="absolute z-10 right-4 top-0"
		size="icon"
		onclick={() => getLocation()}
	>
		<LocateFixed class="w-5 h-5" />
	</Button>
	<div class="flex flex-row gap-2 flex-wrap">
		<DateInput bind:value={time} />
		<RadioGroup.Root
			class="flex"
			bind:value={() => (arriveBy ? 'arrival' : 'departure'), (v) => (arriveBy = v === 'arrival')}
		>
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
		<Button variant="ghost" onclick={() => (expanded = !expanded)}>
			{t.advancedSearchOptions}
			{#if expanded}
				<ChevronUp class="size-[18px]" />
			{:else}
				<ChevronDown class="size-[18px]" />
			{/if}
		</Button>
	</div>
</div>


{#if expanded}
	<div class="w-lg m-4 grid grid-cols-2 items-center space-y-2">
		<!-- Max travel time -->
		<div class="text-sm">
			<!-- TODO -->
			Max travel time
		</div>
		<Select.Root type="single" bind:value={maxTravelTime} items={possibleTravelTimes}>
			<Select.Trigger class="flex items-center w-full overflow-hidden" aria-label="max travel time">
				<div class="w-full text-right pr-4">{maxTravelTime} min</div>
			</Select.Trigger>
			<Select.Content align="end">
				{#each possibleTravelTimes as option, i ( i + option.value )}
					<Select.Item value={option.value} label={option.label}>
						<div class="w-full text-right pr-2">{option.label}</div>
					</Select.Item>
				{/each}
			</Select.Content>
		</Select.Root>

		<!-- First mile -->
		<div class="text-sm">
			{#if arriveBy }
				{t.routingSegments.lastMile}
			{:else}
				{t.routingSegments.firstMile}
			{/if}
		</div>
		<Select.Root type="single" bind:value={oneMileMode}>
			<Select.Trigger
				class="flex items-center w-full overflow-hidden"
				aria-label={arriveBy ? t.routingSegments.lastMile : t.routingSegments.firstMile}
			>
				{t[oneMileMode as TranslationKey]}
			</Select.Trigger>
			<Select.Content sideOffset={10}>
				{#each ['WALK', 'BIKE', 'CAR'] as mode, i (i + mode)}
					<Select.Item value={mode} label={t[mode as TranslationKey] as string}>
						{t[mode as TranslationKey]}
					</Select.Item>
				{/each}
			</Select.Content>
		</Select.Root>

		<!-- Max duration near one location -->
		<div class="text-sm">
			<!-- TODO -->
			Max travel time near one location
		</div>
		<Select.Root type="single" bind:value={maxOneTime} items={possibleTravelTimes}>
			<Select.Trigger class="flex items-center w-full overflow-hidden" aria-label="max travel time">
				<div class="w-full text-right pr-4">{maxOneTime} min</div>
			</Select.Trigger>
			<Select.Content align="end">
				{#each possibleTravelTimes as option, i ( i + option.value )}
					<Select.Item value={option.value} label={option.label}>
						<div class="w-full text-right pr-2">{option.label}</div>
					</Select.Item>
				{/each}
			</Select.Content>
		</Select.Root>

		<!-- Styling -->
		<div class="grid grid-cols-2 items-stretch">
			<div class="text-sm">
				<!-- TODO -->
				Style
			</div>
			<input class="flex right-0 align-right" type="color" bind:value={color} />
		</div>
		<Slider.Root type="single" min={0} max={1000} bind:value={opacity} class="relative flex w-full touch-none select-none items-center">
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
{/if}
