<script lang="ts">
	import circle from "@turf/circle";
	import maplibregl, { type LngLatBoundsLike } from "maplibre-gl";

    interface Pos {
        lat: number;
        lng: number;
        duration: number;
        // name?: string;
    };

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

	const factor = 0.06;  // TODO Calculate factor for duration ~~> radius (in kilometers)
	const box2 = $derived(maplibregl.LngLatBounds.convert(bounds ? bounds : [[0, 0], [0, 0]]));
    function reachable_kilemeters(pos: Pos) {
        return pos.duration * factor;
    }
    function is_visible(data: Pos) {
        const r = reachable_kilemeters(data);
        return box2._sw.lat <= data.lat + r && data.lat - r <= box2._ne.lat
			&& box2._sw.lng <= data.lng + r && data.lng - r <= box2._ne.lat;
    }
	function transform(p: number[]) {
		return [p[0] - box2._sw.lng, box2._ne.lat - p[1]]
	}
	const circles2 = $derived(
		isochronesData
			.filter(is_visible)
            .map((data) => {
                const r = reachable_kilemeters(data);
                const c = circle(
                    [data.lng, data.lat],
                    r,
                    {
                        // steps: steps,
                        // steps: 10,
                        // steps: 3,
                        units: "kilometers",
                        properties: {
                            // "name": data.name ?? 'unknown',
                        }
                    }
                );
				const coords = c.geometry.coordinates[0].map(transform);
				// console.log(coords);
				// console.log(c.geometry.coordinates);
				// const x = c.geometry.coordinates.map(p => p.map(q => q.toString())).join(' ');
				// const x = c.geometry.coordinates.map(p => p.join(',')).join(' ') + ' ' + c.geometry.coordinates[0][0] + ',' + c.geometry.coordinates[0][1];
				// const x = c.geometry.coordinates[0].join(' ') + ' ' + c.geometry.coordinates[0][0];
				const x = coords.join(' ') + ' ' + coords[0];
				// console.log("x:", x);
				return x;
				// return c;
            })
    );

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
	{#each circles2 as circle}
		<polygon points="{circle}" style="fill:green" />
		<!-- <circle cx="{circle.cx}" cy="{circle.cy}" r="{circle.r}" fill="green" /> -->
	{/each}
</svg>
