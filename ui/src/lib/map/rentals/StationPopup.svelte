<script lang="ts">
	import type {
		RentalFormFactor,
		RentalProvider,
		RentalStation
	} from '@motis-project/motis-client';
	import { Button } from '$lib/components/ui/button';
	import { Copy, type Icon as IconType } from '@lucide/svelte';
	import { formFactorAssets, propulsionTypes, returnConstraints } from '$lib/map/rentals/assets';
	import { t } from '$lib/i18n/translation';

	let {
		provider,
		station,
		showActions = false,
		debug = false
	}: {
		provider: RentalProvider;
		station: RentalStation;
		showActions?: boolean;
		debug?: boolean;
	} = $props();

	let debugInfo = $derived({
		station,
		provider: {
			...provider,
			vehicleTypes: provider.vehicleTypes.filter(
				(vt) =>
					Object.hasOwn(station.vehicleTypesAvailable, vt.id) ||
					Object.hasOwn(station.vehicleDocksAvailable, vt.id)
			),
			totalVehicleTypes: provider.vehicleTypes.length
		}
	});

	async function copyDebugInfo() {
		await navigator.clipboard.writeText(JSON.stringify(debugInfo, null, 2));
	}

	type IconInfo = {
		component: typeof IconType;
		title: string;
	};

	type VehicleRow = {
		id: string;
		available: number;
		formFactor: RentalFormFactor;
		name: string;
		propulsionIcon: IconInfo | null;
		returnIcon: IconInfo | null;
	};

	const vehicleRows = $derived.by<VehicleRow[]>(() => {
		return Object.entries(station.vehicleTypesAvailable)
			.map(([id, count]) => {
				const vt = provider.vehicleTypes.find((vt) => vt.id === id)!;
				const name = vt.name || formFactorAssets[vt.formFactor].label;

				return {
					id,
					available: count,
					formFactor: vt.formFactor,
					name,
					propulsionIcon: propulsionTypes[vt.propulsionType],
					returnIcon: returnConstraints[vt.returnConstraint]
				};
			})
			.sort((a, b) => {
				if (b.available !== a.available) {
					return b.available - a.available;
				}
				return a.name.localeCompare(b.name);
			});
	});
</script>

<div class="space-y-3 text-sm leading-tight text-foreground max-w-96 w-fit">
	<div class="space-y-1">
		<div class="font-semibold">{station.name}</div>
		{#if station.address}
			<div>{station.address}</div>
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
	{#if vehicleRows.length}
		<table class="w-full text-xs">
			<tbody>
				{#each vehicleRows as vehicle (vehicle.id)}
					<tr class="border-b border-border last:border-0">
						<td class="w-8 pr-2 align-middle">
							{vehicle.available}x
						</td>
						<td class="w-6 pr-2 align-middle">
							<svg class="h-4 w-4 fill-current" aria-hidden="true" focusable="false">
								<title>{formFactorAssets[vehicle.formFactor].label}</title>
								<use href={`#${formFactorAssets[vehicle.formFactor].svg}`} />
							</svg>
						</td>
						<td class="w-6 pr-2 align-middle">
							{#if vehicle.propulsionIcon}
								{@const PropulsionIcon = vehicle.propulsionIcon.component}
								<span
									class="inline-flex h-4 w-4 items-center justify-center text-muted-foreground"
									role="img"
									title={vehicle.propulsionIcon.title}
									aria-label={vehicle.propulsionIcon.title}
								>
									<PropulsionIcon class="h-4 w-4" aria-hidden="true" />
								</span>
							{/if}
						</td>
						<td
							class="truncate align-middle max-w-64"
							title={vehicle.name}
							aria-label={vehicle.name}
						>
							{vehicle.name}
						</td>
						<td class="w-8 pl-2 align-middle text-right">
							{#if vehicle.returnIcon}
								{@const ReturnIcon = vehicle.returnIcon.component}
								<span
									class="inline-flex h-4 w-4 items-center justify-center text-muted-foreground"
									role="img"
									title={vehicle.returnIcon.title}
									aria-label={vehicle.returnIcon.title}
								>
									<ReturnIcon class="h-4 w-4" aria-hidden="true" />
								</span>
							{/if}
						</td>
					</tr>
				{/each}
			</tbody>
		</table>
	{/if}
	{#if showActions && station.rentalUriWeb}
		<Button class="font-bold" variant="outline" href={station.rentalUriWeb} target="_blank">
			{t.rent}
		</Button>
	{/if}
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
