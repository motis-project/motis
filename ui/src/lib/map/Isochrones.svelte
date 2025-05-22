<script lang="ts">
	import circle from "@turf/circle";
	import { validateBBox } from "@turf/helpers";
	import maplibregl, { CanvasSource, type LngLatBoundsLike, Map } from "maplibre-gl";

    interface Pos {
        lat: number;
        lng: number;
        duration: number;
        // name?: string;
    };

	let {
        map = $bindable(),
		bounds,
		// class: user_class,
		isochronesData
	} : {
        map: Map | undefined;
		bounds: LngLatBoundsLike | undefined;
		// class: string;
		// isochronesData: {center: string, duration: number}[]
		isochronesData: Pos[];
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

	const name = 'isochrones-data';
    let loaded = false;

	const factor = 0.06;  // TODO Calculate factor for duration ~~> radius (in kilometers)
	const box2 = $derived(maplibregl.LngLatBounds.convert(bounds ? bounds : [[0, 0], [1, 1]]));
	const box_coords: [[number, number], [number, number], [number, number], [number, number]] = $derived(
		[
			[box2._sw.lng, box2._ne.lat],
			[box2._ne.lng, box2._ne.lat],
			[box2._ne.lng, box2._sw.lat],
			[box2._sw.lng, box2._sw.lat],
		]
	);
    function reachable_kilemeters(pos: Pos) {
        return pos.duration * factor;
    }
	// const dimensions = $derived(map?._containerDimensions());
    function is_visible(data: Pos) {
        return box2._sw.lat <= data.lat && data.lat <= box2._ne.lat
			&& box2._sw.lng <= data.lng && data.lng <= box2._ne.lat;
        // const r = reachable_kilemeters(data);
        // return box2._sw.lat <= data.lat + r && data.lat - r <= box2._ne.lat
		// 	&& box2._sw.lng <= data.lng + r && data.lng - r <= box2._ne.lat;
    }
	// function transform(p: number[]) {
	// 	return [p[0] - box2._sw.lng, box2._ne.lat - p[1]]
	// }
	function transform(pos: number[], dimensions: number[]) {
		const x = Math.round(((pos[0] - box2._sw.lng) / (box2._ne.lng - box2._sw.lng)) * dimensions[0]);
		const y = Math.round(((box2._ne.lat - pos[1]) / (box2._ne.lat - box2._sw.lat)) * dimensions[1]);
		// console.log("T: ", [x, y],"     (", pos, ")");
		return [x, y];
	}
	const circles2 = $derived(
		isochronesData
			// .filter(is_visible)
            .map((data) => {
                const r = reachable_kilemeters(data);
                const c = circle(
                    [data.lng, data.lat],
                    r,
                    {
                        // steps: steps,
                        // steps: 10,
                        steps: 3,
                        units: "kilometers",
                        properties: {
                            // "name": data.name ?? 'unknown',
                        }
                    }
                );
				// const coords = c.geometry.coordinates[0];
				// const x = coords.join(' ') + ' ' + coords[0];
				// console.log("x:", x);
				// return x;
				return c;
            })
    );

	$effect(() => {
		if(!map) {
			return;
		}
        if (!loaded) {
            map.addSource(
				name,
				{
					type: "canvas",
					canvas: "isochronesCanvas",
					"coordinates": box_coords,
					// coordinates: [
					// 	[box2._sw.lng, box2._ne.lat],
					// 	[box2._ne.lng, box2._ne.lat],
					// 	[box2._ne.lng, box2._sw.lat],
					// 	[box2._sw.lng, box2._sw.lat],
					// ]
				}
			);
            map.addLayer({
                id: name,
                type: "raster",
                source: name,
                paint: {
					// "raster-opacity": 0.25,
                }
            });
            loaded = true;
        }

		const dimensions = map._containerDimensions();
		console.log('Dimensions: ', dimensions);
		const source = (map.getSource(name) as CanvasSource);
		source.setCoordinates(box_coords);

		const canvas = source.canvas;
		canvas.width = dimensions[0];
		canvas.height = dimensions[1];
		const ctx = canvas.getContext("2d");
		if (!ctx) {
			return;
		}
		// ctx.fillStyle = "magenta";
		// ctx.fillRect(600, 600, 300, 300);
		console.log("BOX: ", box2);

		circles2.forEach(c => {
			ctx.save();
			ctx.beginPath();
			// ctx.fillStyle = "red";
			// ctx.fill();

			// Clip circle
			ctx.strokeStyle = "red";
			const coords = c.geometry.coordinates[0];
			const start = transform(coords[0], dimensions);
			// const start = coords[0];
			ctx.moveTo(start[0], start[1]);
			for (let i = 0; i < coords.length; ++i) {
				// const pos = coords[i];
				const pos = transform(coords[i], dimensions);
				ctx.lineTo(pos[0], pos[1]);
			}
			ctx.clip();
			// ctx.stroke();

			// Fill map, clipped to circle
			ctx.fillStyle = "yellow";
			// console.log("S:", start);
			ctx.fillRect(0, 0, dimensions[0], dimensions[1]);
			// ctx.fillRect(start[0], start[1], 10, 10);
			// c.geometry.coordinates.sl(p => {
			// 	p.
			// })
			// const b = c.geometry.bbox;
			// console.log(`BOX: ${b}`, c);
			ctx.restore();
		});
			// ctx.clip();
			// ctx.fillStyle = "yellow";
			// ctx.fillRect(0, 0, dimensions[0], dimensions[1]);
		// const size = getElementSize
		// const width =
	});

	// const circles = $derived.by(() => {
	// 	if (bounds === undefined) {
	// 		console.log('NO VIEW');
	// 		return [];
	// 	}
	// 	const factor = 0.0004;  // TODO Calculate factor for duration ~~> radius
	// 	let box = maplibregl.LngLatBounds.convert(bounds);
	// 	const c = isochronesData
	// 		.filter((data) => box._sw.lat <= data.lat && data.lat <= box._ne.lat
	// 		&& box._sw.lng <= data.lng && data.lng <= box._ne.lat)
	// 	.map((data) => {
	// 	// Flip y coordinate
	// 	const c = {
	// 		cx: data.lng - box._sw.lng,
	// 		cy: box._ne.lat - data.lat,
	// 		r: data.duration * factor
	// 	};
	// 	console.log(`CIRCLE: ${JSON.stringify(c)}`);
	// 	return c;
	// })
	// ;
	// console.log(`ALL CIRCLES: ${JSON.stringify(c)}`);
	// return c;
	// }
	// );
</script>

<svg class="opacity-25 pointer-events-none {user_class}" viewBox="{box}" preserveAspectRatio="none">
	{#each circles2 as circle}
		<polygon points="{circle}" style="fill:green" />
		<!-- <circle cx="{circle.cx}" cy="{circle.cy}" r="{circle.r}" fill="green" /> -->
	{/each}
</svg>
