<script lang="ts">
	import type { RentalProvider, RentalVehicle } from '$lib/api/openapi';
	import { Button } from '$lib/components/ui/button';
	import { t } from '$lib/i18n/translation';

	let {
		provider,
		vehicle,
		showActions = false
	}: {
		provider: RentalProvider;
		vehicle: RentalVehicle;
		showActions?: boolean;
	} = $props();
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
</div>
