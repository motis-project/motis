<script lang="ts">
	import AddressTypeahead from '$lib/AddressTypeahead.svelte';
	import { type Location } from '$lib/Location';
	import { t } from '$lib/i18n/translation';
	import { onClickStop } from '$lib/utils';
	import { page } from '$app/state';
	import DateInput from '$lib/DateInput.svelte';

	let time = $derived(page.state.selectedStop?.time ?? new Date(Date.now()));
	let from = $state<Location>() as Location;
	let fromItems = $state<Array<Location>>([]);
	const refreshStops = (location?: Location) => {
		let selectedStop =
			location && location.match
				? { label: location.label, id: location.match.id }
				: page.state.selectedStop
					? { label: page.state.selectedStop.name, id: page.state.selectedStop.stopId }
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
	<div class="flex min-h-0 flex-row gap-2 flex-wrap">
		<DateInput
			bind:value={
				() => time,
				(v) => {
					// Using $effect on time to refresh the stops is not possible
					// because it leads to an infinite effect-update loop,
					// so we refresh the stops in the setter part of bind
					time = v;
					refreshStops();
				}
			}
		/>
	</div>
</div>
