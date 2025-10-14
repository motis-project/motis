<script lang="ts">
	import type { RentalProvider, RentalZone } from '$lib/api/openapi';
	import { t } from '$lib/i18n/translation';

	let {
		provider,
		zone,
		rideThroughAllowed,
		rideEndAllowed
	}: {
		provider: RentalProvider;
		zone: RentalZone;
		rideThroughAllowed: boolean;
		rideEndAllowed: boolean;
	} = $props();
</script>

<div class="space-y-3 text-sm leading-tight text-foreground">
	<div class="space-y-1">
		{#if zone.name}
			<div class="font-semibold">{zone.name}</div>
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
		<div>
			{rideEndAllowed ? t.rideEndAllowed : t.rideEndNotAllowed}
		</div>
	</div>
</div>
