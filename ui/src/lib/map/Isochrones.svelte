<script lang="ts">
	import maplibregl, { type LngLatBoundsLike } from "maplibre-gl";

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
		const bv = `0 0 ${w} ${h}`;
		console.log(`BOX VIEW: ${bv}  –  ${b}`);
		return bv;
	});
	const circles = $derived.by(() => {
		if (bounds === undefined) {
			console.log('NO VIEW');
			return [];
		}
		const factor = 0.0004;  // TODO Calculate factor for duration ~~> radius
		let box = maplibregl.LngLatBounds.convert(bounds);
		const c = isochronesData
			.filter((data) => box._sw.lat <= data.lat && data.lat <= box._ne.lat
			&& box._sw.lng <= data.lng && data.lng <= box._ne.lat)
		.map((data) => {
		// Flip y coordinate
		const c = {
			cx: data.lng - box._sw.lng,
			cy: box._ne.lat - data.lat,
			r: data.duration * factor
		};
		console.log(`CIRCLE: ${JSON.stringify(c)}`);
		return c;
	})
	;
	console.log(`ALL CIRCLES: ${JSON.stringify(c)}`);
	return c;
	}
	);
</script>

<svg class="opacity-25 pointer-events-none {user_class}" viewBox="{box}" preserveAspectRatio="none">
	{#each circles as circle}
		<circle cx="{circle.cx}" cy="{circle.cy}" r="{circle.r}" fill="green" />
	{/each}
</svg>
