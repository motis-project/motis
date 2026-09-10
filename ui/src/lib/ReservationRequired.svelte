<script lang="ts">
	import { Icon } from '@lucide/svelte';

	const iconNode: [string, Record<string, string>][] = [
		['circle', { cx: '12', cy: '12', r: '10' }],
		['path', { d: 'M9 16.5V7.5' }],
		['path', { d: 'M9 7.5h3.25a2.75 2.75 0 0 1 0 5.5H9' }],
		['path', { d: 'M11.75 13 15 16.5' }]
	];

	let props = $props();
</script>

<Icon name="reservation-required" {...props} {iconNode} />
