<script lang="ts">
	import { stoptimes, type StoptimesResponse } from '$lib/openapi';
	import LoaderCircle from 'lucide-svelte/icons/loader-circle';
	import ArrowRight from 'lucide-svelte/icons/arrow-right';
	import Time from '$lib/Time.svelte';
	import Route from '$lib/Route.svelte';
	import { Button } from '$lib/components/ui/button';

	let {
		stopId,
		time: queryTime,
		arriveBy = $bindable(),
		onClickTrip
	}: {
		stopId: string;
		time: Date;
		arriveBy?: boolean;
		onClickTrip: (tripId: string, date: string) => void;
	} = $props();

	let query = $derived({ stopId, time: queryTime.toISOString(), arriveBy, n: 10 });
	let responses = $state<Array<Promise<StoptimesResponse>>>([]);
	$effect(() => {
		responses = [stoptimes({ query }).then((r) => r.data!)];
	});
</script>

<div class="text-base grid gap-y-2 grid-cols-10 auto-rows-fr items-center">
	<div class="col-span-full w-full flex items-center justify-center">
		<Button
			class="font-bold"
			variant="outline"
			on:click={() => {
				arriveBy = !arriveBy;
			}}
		>
			{#if arriveBy}
				Wechsel zu Abfahrten
			{:else}
				Wechsel zu Ankünften
			{/if}
		</Button>
	</div>
	{#each responses as r, rI}
		{#await r}
			<div class="col-span-full w-full flex items-center justify-center">
				<LoaderCircle class="animate-spin w-12 h-12 m-20" />
			</div>
		{:then r}
			{#if rI === 0}
				<div class="col-span-full w-full flex justify-between items-center space-x-4">
					<div class="border-t w-full h-0"></div>
					<button
						onclick={() => {
							responses.splice(
								0,
								0,
								stoptimes({ query: { ...query, pageCursor: r.previousPageCursor } }).then(
									(x) => x.data!
								)
							);
						}}
						class="px-2 py-1 bg-blue-600 hover:!bg-blue-700 text-white font-bold text-sm border rounded-lg"
					>
						früher
					</button>
					<div class="border-t w-full h-0"></div>
				</div>
			{/if}

			{#each r.stopTimes as t}
				{@const time = arriveBy ? t.place.arrival! : t.place.departure!}
				{@const delay = arriveBy ? t.place.arrivalDelay! : t.place.departureDelay!}
				<Route
					class="col-span-3 w-fit max-w-28 text-ellipsis overflow-hidden"
					l={t}
					{onClickTrip}
				/>
				<Time class="ml-1" rt={false} isRealtime={t.realTime} timestamp={time} {delay} />
				<Time class="ml-2" rt={true} isRealtime={t.realTime} timestamp={time} {delay} />
				<div class="ml-4 col-span-5 flex items-center text-muted-foreground">
					<div><ArrowRight class="stroke-muted-foreground h-4 w-4" /></div>
					<span class="ml-1 text-nowrap text-ellipsis overflow-hidden">{t.headsign}</span>
				</div>
			{/each}

			{#if rI === responses.length - 1}
				<div class="col-span-full w-full flex justify-between items-center space-x-4">
					<div class="border-t w-full h-0"></div>
					<button
						onclick={() => {
							responses.push(
								stoptimes({ query: { ...query, pageCursor: r.nextPageCursor } }).then(
									(x) => x.data!
								)
							);
						}}
						class="px-2 py-1 bg-blue-600 hover:!bg-blue-700 text-white text-sm font-bold border rounded-lg"
					>
						später
					</button>
					<div class="border-t w-full h-0"></div>
				</div>
			{/if}
		{/await}
	{/each}
</div>
