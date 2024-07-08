<script lang="ts">
	let { timestamp, delay, ...props }: { timestamp: number; delay?: number } = $props();

	const d = new Date(timestamp + (delay || 0));
	const pad = (x: number) => ('0' + x).slice(-2);
</script>

<div class="w-16" {...props}>
	{#if delay === undefined || (delay !== undefined && delay !== 0)}
		{pad(d.getHours())}:{pad(d.getMinutes())}
	{/if}
</div>
