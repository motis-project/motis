<script lang="ts">
	import circle from "@turf/circle";
	import { validateBBox } from "@turf/helpers";
	import maplibregl, { CanvasSource, type LngLatBoundsLike, Map } from "maplibre-gl";

    interface Pos {
        lat: number;
        lng: number;
        duration: number;
    };

	let {
        map = $bindable(),
		bounds,
		isochronesData
	} : {
        map: Map | undefined;
		bounds: LngLatBoundsLike | undefined;
		isochronesData: Pos[];
	} = $props();

	const name = 'isochrones-data';
    let loaded = false;

	const factor = 0.06;  // 3.6 kilometers per hour
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
    // function is_visible(data: Pos) {
    //     return box2._sw.lat <= data.lat && data.lat <= box2._ne.lat
	// 		&& box2._sw.lng <= data.lng && data.lng <= box2._ne.lat;
    // }
	function transform(pos: number[], dimensions: number[]) {
		const x = Math.round(((pos[0] - box2._sw.lng) / (box2._ne.lng - box2._sw.lng)) * dimensions[0]);
		const y = Math.round(((box2._ne.lat - pos[1]) / (box2._ne.lat - box2._sw.lat)) * dimensions[1]);
		return [x, y];
	}
	const circles2 = $derived(
		isochronesData
            .map((data) => {
                const r = reachable_kilemeters(data);
                const c = circle(
                    [data.lng, data.lat],
                    r,
                    {
						// steps: 64,
                        units: "kilometers",
                    }
                );
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
				}
			);
            map.addLayer({
                id: name,
                type: "raster",
                source: name,
                paint: {
					"raster-opacity": 0.25,
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
		ctx.fillStyle = "yellow";
		ctx.clearRect(0, 0, dimensions[0], dimensions[1]);

		circles2.forEach(c => {
			ctx.save();  // Store canvas state

			// Clip circle
			ctx.beginPath();
			const coords = c.geometry.coordinates[0];
			const start = transform(coords[0], dimensions);
			ctx.moveTo(start[0], start[1]);
			for (let i = 0; i < coords.length; ++i) {
				const pos = transform(coords[i], dimensions);
				ctx.lineTo(pos[0], pos[1]);
			}
			ctx.clip();

			// Fill map, clipped to circle
			ctx.fillRect(0, 0, dimensions[0], dimensions[1]);

			// Restore previous state on top
			ctx.restore();
		});
	});
</script>

<svg class="opacity-25 pointer-events-none {user_class}" viewBox="{box}" preserveAspectRatio="none">
	{#each circles2 as circle}
		<polygon points="{circle}" style="fill:green" />
		<!-- <circle cx="{circle.cx}" cy="{circle.cy}" r="{circle.r}" fill="green" /> -->
	{/each}
</svg>
