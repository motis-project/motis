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
	const refreshStops = (location?: Location) => {
		let selectedStop =
			location && location.match
				? { label: location.label, id: location.match.id }
				: from && from.match
					? { label: from.label, id: from.match.id }
					: null;
		if (selectedStop) {
			onClickStop(selectedStop.label, selectedStop.id, time);
		}
	};
</script>

<div id="searchmask-container" class="flex flex-col space-y-4 p-4 relative">
	<AddressTypeahead
		name="from"
		placeholder={t.from}
		bind:selected={from}
		bind:items={fromItems}
		type="STOP"
		onChange={refreshStops}
	/>
</div>
