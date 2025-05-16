<script lang="ts">
	// import CircleX from "lucide-svelte/icons/circle-x";
	import maplibregl, { type LngLatBoundsLike } from "maplibre-gl";
	import { BorderWidth } from "svelte-radix";

	let {
		bounds,
		class: user_class,
		isochronesData
	} : {
		bounds: LngLatBoundsLike | undefined;
		class: string;
		// isochronesData: {center: string, duration: number}[]
		isochronesData: {lat: number, lng: number, duration: number}[]
		} = $props();

	const box = $derived.by(() => {
		if (bounds === undefined) {
			console.log('MIN BOX');
			return "0 0 1 1";
		}
		let b = maplibregl.LngLatBounds.convert(bounds);
		const w = b._ne.lng - b._sw.lng;
		const h = b._ne.lat - b._sw.lat;
		const bv = `${b._sw.lng} ${b._sw.lat} ${w} ${h}`;
		console.log(`BOX VIEW: ${bv}`);
		return bv;
	});
	const circles = $derived.by(() => {
		if (bounds === undefined) {
			console.log('NO VIEW');
			return [];
		}
		const factor = 0.0004;  // TODO Calculate factor for duration ~~> radius
		let box = maplibregl.LngLatBounds.convert(bounds);
		const h = box._ne.lat + box._sw.lat;
		const c = isochronesData
			.filter((data) => box._sw.lat <= data.lat && data.lat <= box._ne.lat
			&& box._sw.lng <= data.lng && data.lng <= box._ne.lat)
		.map((data) => {
		// const pos =
		// const c = {cx: data.lng, cy: data.lat, r: data.duration * factor};
		const c = {cx: data.lng, cy: h - data.lat, r: data.duration * factor};
		console.log(`CIRCLE: ${JSON.stringify(c)}`);
		return c;
	})
	//.filter((d) => d != null)
	;
	console.log(`ALL CIRCLES: ${JSON.stringify(c)}`);
	return c;
	}
	);
</script>

<svg class="opacity-25 pointer-events-none {user_class}" viewBox="{box}">
	{#each circles as circle}
	<!-- {console.log(`Adding circle: ${circle}`)} -->
		<circle cx="{circle.cx}" cy="{circle.cy}" r="{circle.r}" fill="green" />
	{/each}
	<!-- <circle cx="30" cy="50" r="30" fill="red" /> -->
	<!-- <circle cx="60" cy="40" r="20" fill="blue" /> -->
</svg>
