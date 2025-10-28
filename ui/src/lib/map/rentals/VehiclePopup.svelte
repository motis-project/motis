<script lang="ts">
	import type { RentalProvider, RentalVehicle } from '$lib/api/openapi';
	import { Button } from '$lib/components/ui/button';
	import Copy from 'lucide-svelte/icons/copy';
	import { t } from '$lib/i18n/translation';

	let {
		provider,
		vehicle,
		showActions = false,
		debug = false
	}: {
		provider: RentalProvider;
		vehicle: RentalVehicle;
		showActions?: boolean;
		debug?: boolean;
	} = $props();

	let debugInfo = $derived({
		vehicle,
		provider: {
			...provider,
			vehicleTypes: provider.vehicleTypes.filter((vt) => vt.id == vehicle.typeId),
			totalVehicleTypes: provider.vehicleTypes.length
		}
	});

	async function copyDebugInfo() {
		await navigator.clipboard.writeText(JSON.stringify(debugInfo, null, 2));
	}
</script>

<div class="space-y-3 text-sm leading-tight text-foreground">
	<div>
		{t.sharingProvider}:
		{#if provider.url}
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
	{#if showActions && vehicle.rentalUriWeb}
		<Button class="font-bold" variant="outline" href={vehicle.rentalUriWeb} target="_blank">
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
