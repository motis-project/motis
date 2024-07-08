<script lang="ts">
	let { timestamp }: { timestamp: number } = $props();

	const d = new Date(timestamp);
	const pad = (x: number) => ('0' + x).slice(-2);
</script>

{pad(d.getHours())}:{pad(d.getMinutes())}
