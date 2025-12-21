<script lang="ts">
	import AddressTypeahead from '$lib/AddressTypeahead.svelte';
	import { type Location } from '$lib/Location';
	import { t } from '$lib/i18n/translation';
	import { onClickStop } from '$lib/utils';

	let {
		time = $bindable()
	}: {
		time: Date;
	} = $props();

	let from = $state<Location>() as Location;
	let fromItems = $state<Array<Location>>([]);
</script>

<div id="searchmask-container" class="flex flex-col space-y-4 p-4 relative">
	<AddressTypeahead
		name="from"
		placeholder={t.from}
		bind:selected={from}
		bind:items={fromItems}
		type="STOP"
		onChange={(location) => {
			if (location.match) {
				onClickStop(location.label, location.match.id, time);
			}
		}}
	/>
</div>
