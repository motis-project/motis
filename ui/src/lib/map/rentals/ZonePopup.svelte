<script lang="ts">
	import type { RentalProvider, RentalStation, RentalZone } from '@motis-project/motis-client';
	import { Button } from '$lib/components/ui/button';
	import { Copy } from '@lucide/svelte';
	import { t } from '$lib/i18n/translation';
	import type { RentalZoneFeature } from './zone-types';

	let {
		provider,
		zone,
		station,
		rideThroughAllowed,
		rideEndAllowed,
		allZonesAtPoint = [],
		zoneData = [],
		stationData = [],
		debug = false
	}: {
		provider: RentalProvider;
		zone: RentalZone | undefined;
		station: RentalStation | undefined;
		rideThroughAllowed: boolean;
		rideEndAllowed: boolean;
		allZonesAtPoint?: RentalZoneFeature[];
		zoneData?: RentalZone[];
		stationData?: RentalStation[];
		debug?: boolean;
	} = $props();

	let debugInfo = $derived({
		...(zone ? { zone: { ...zone, area: '...' } } : {}),
		...(station ? { station: { ...station, stationArea: '...' } } : {}),
		provider: provider,
		allZonesAtPoint: allZonesAtPoint.map((feature) => ({
			z: feature.properties.z,
			zoneIndex: feature.properties.zoneIndex,
			stationIndex: feature.properties.stationIndex,
			rideThroughAllowed: feature.properties.rideThroughAllowed,
			rideEndAllowed: feature.properties.rideEndAllowed,
			stationArea: feature.properties.stationArea,
			zone: (() => {
				if (feature.properties.zoneIndex === undefined) {
					return null;
				}
				const zone = zoneData[feature.properties.zoneIndex];
				return zone ? { ...zone, area: '...' } : null;
			})(),
			station: (() => {
				if (feature.properties.stationIndex === undefined) {
					return null;
				}
				const station = stationData[feature.properties.stationIndex];
				return station ? { ...station, stationArea: '...' } : null;
			})()
		}))
	});

	async function copyDebugInfo() {
		await navigator.clipboard.writeText(JSON.stringify(debugInfo, null, 2));
	}
</script>

<div class="space-y-3 text-sm leading-tight text-foreground">
	<div class="space-y-1">
		{#if zone}
			{#if zone.name}
				<div class="font-semibold">{t.rentalGeofencingZone}: {zone.name}</div>
			{:else}
				<div class="font-semibold">{t.rentalGeofencingZone}</div>
			{/if}
		{:else if station}
			{#if station.name}
				<div class="font-semibold">{t.rentalStation}: {station.name}</div>
			{:else}
				<div class="font-semibold">{t.rentalStation}</div>
			{/if}
		{/if}
		<div>
			{t.sharingProvider}: {#if provider.url}
				<a
					href={provider.url}
					target="_blank"
					class="text-blue-600 dark:text-blue-300 hover:underline"
				>
					{provider.name}
				</a>
			{:else}
				{provider.name}
			{/if}
		</div>
	</div>
	<div class="space-y-1">
		<div>
			{rideThroughAllowed ? t.rideThroughAllowed : t.rideThroughNotAllowed}
		</div>
		{#if rideThroughAllowed}
			<div>
				{rideEndAllowed ? t.rideEndAllowed : t.rideEndNotAllowed}
			</div>
		{/if}
	</div>
	{#if debug}
		<div
			class="pt-2 border-t border-border text-xs text-muted-foreground space-y-1 max-h-96 max-w-96 overflow-auto pr-2 relative"
		>
			<Button
				class="absolute top-2 right-2 z-10"
				variant="ghost"
				size="icon"
				onclick={copyDebugInfo}
				type="button"
				title={t.copyToClipboard}
				aria-label={t.copyToClipboard}
			>
				<Copy />
			</Button>
			<pre class="whitespace-pre-wrap pr-8">{JSON.stringify(debugInfo, null, 2)}</pre>
		</div>
	{/if}
</div>
