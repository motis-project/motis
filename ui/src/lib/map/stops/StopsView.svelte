<script lang="ts">
	import { t } from '$lib/i18n/translation';
	import { lngLatToStr } from '$lib/lngLatToStr';
	import { MapboxOverlay } from '@deck.gl/mapbox';
	import { IconLayer } from '@deck.gl/layers';
	import maplibregl, { type LngLatLike } from 'maplibre-gl';
	import { onMount, untrack } from 'svelte';
	import { stops, type Mode } from '@motis-project/motis-client';
	import { type PickingInfo } from '@deck.gl/core';
	import { onClickStop } from '$lib/utils';
	import { updateOverlayLayers } from '$lib/updateOverlay';
	import { createStopIcon } from '../createIcon';
	import { hexToRgb } from '$lib/Color';
	import { getModeStyle, type LegLike } from '$lib/modeStyle';

	let {
		map,
		overlay,
		layers,
		zoom,
		stopsMode,
		bounds
	}: {
		map: maplibregl.Map | undefined;
		overlay: MapboxOverlay;
		layers: IconLayer[];
		zoom: number;
		stopsMode: 'none' | 'all' | 'grouped';
		bounds: maplibregl.LngLatBoundsLike | undefined;
	} = $props();

	//QUERY
	let query = $derived.by(() => {
		if (!bounds) return null;
		const b = maplibregl.LngLatBounds.convert(bounds);
		const max = lngLatToStr(b.getNorthWest());
		const min = lngLatToStr(b.getSouthEast());
		const grouped = stopsMode == 'grouped';
		let modes: Mode[] | undefined = [];
		if (zoom > 7) {
			modes.push('HIGHSPEED_RAIL');
			modes.push('LONG_DISTANCE');
		}
		if (zoom > 11) {
			modes.push('REGIONAL_RAIL');
			modes.push('ODM');
		}
		if (zoom > 13) {
			modes.push('SUBWAY');
			modes.push('SUBURBAN');
		}
		if (zoom > 15) {
			modes.push('BUS');
			modes.push('TRAM');
		}
		if (zoom > 17) {
			modes = undefined;
		}
		return {
			min,
			max,
			grouped,
			modes
		};
	});

	//DATA
	const STOPS_NUM = 2048;
	const colors = new Uint8Array(STOPS_NUM * 3);
	const positions = new Float64Array(STOPS_NUM * 2);
	const stopsData = {
		length: STOPS_NUM,
		positions,
		colors
	};
	type MetaData = {
		name: string;
		stopId?: string;
		track?: string;
		modes?: Array<Mode>;
	};
	const metadata: MetaData[] = [];

	//LAYER
	const ICON_SIZE = 50;
	const StopIcon = createStopIcon(ICON_SIZE);

	const IconMapping = {
		marker: {
			x: 0,
			y: 0,
			width: ICON_SIZE,
			height: ICON_SIZE,
			anchorX: ICON_SIZE / 2,
			anchorY: ICON_SIZE / 2,
			mask: true
		}
	};

	const popup = new maplibregl.Popup({
		closeButton: false,
		closeOnClick: false,
		maxWidth: 'none'
	});

	const createModeIcon = (mode: Mode): HTMLElement => {
		const [icon, bg, fg] = getModeStyle({ mode } as LegLike);
		const span = Object.assign(document.createElement('span'), {
			innerHTML: `<svg width="17" height="17" fill="currentColor"><use href="#${icon}"></use></svg>`
		});
		Object.assign(span.style, {
			display: 'flex',
			alignItems: 'center',
			gap: '4px',
			background: bg,
			color: fg,
			borderRadius: '8px',
			padding: '4px'
		});
		return span;
	};

	const onHover = (info: PickingInfo) => {
		if (!info.picked || info.index === -1) {
			popup.remove();
			return;
		}
		const data = metadata[info.index];

		const root = document.createElement('div');
		Object.assign(root.style, {
			display: 'flex',
			alignItems: 'center',
			flexDirection: 'column',
			gap: '6px',
			marginRight: '-28px'
		});

		const name = Object.assign(document.createElement('span'), { textContent: data.name });
		Object.assign(name.style, {
			fontSize: '14px',
			fontWeight: '700',
			lineHeight: '1.2'
		});
		root.appendChild(name);

		if (data.track) {
			const track = document.createElement('span');
			Object.assign(track.style, {
				background: 'rgba(0,0,0,0.08)',
				borderRadius: '6px',
				padding: '5px',
				fontSize: '12px',
				fontWeight: '700'
			});
			track.innerHTML = `${t.track}: ${data.track}`;
			root.appendChild(track);
		}

		if (data.modes?.length) {
			const modeRow = document.createElement('div');
			Object.assign(modeRow.style, {
				display: 'flex',
				flexWrap: 'wrap',
				gap: '4px'
			});
			data.modes.forEach((m) => modeRow.appendChild(createModeIcon(m)));
			root.appendChild(modeRow);
		}

		popup
			.setLngLat(info.coordinate as LngLatLike)
			.setDOMContent(root)
			.addTo(map!);
	};

	const onClick = (info: PickingInfo) => {
		if (info.picked && info.index != -1) {
			const data = metadata[info.index];
			onClickStop(data.name, data.stopId!, new Date(Date.now()));
		}
	};

	const createLayer = () => {
		return new IconLayer({
			id: 'stops-view-layer',
			beforeId: 'trips-layer',
			data: {
				length: stopsData.length,
				attributes: {
					getPosition: { value: positions, size: 2 },
					getColor: { value: colors, size: 3, normalized: true }
				}
			},
			// @ts-expect-error: canvas element seems to work fine
			iconAtlas: StopIcon,
			iconMapping: IconMapping,
			getSize: 15,
			pickable: stopsMode !== 'none',
			colorFormat: 'RGB',
			visible: stopsMode !== 'none',
			useDevicePixels: false,
			autoHighlight: true,
			parameters: { depthTest: false },
			getIcon: (_) => 'marker',
			onHover,
			onClick
		});
	};

	//SETUP
	onMount(() => {
		updateOverlayLayers(createLayer(), layers, overlay);
	});

	//UPDATE
	$effect(() => {
		if (stopsMode) {
			updateOverlayLayers(createLayer(), layers, overlay);
			stopsData.length = 0;
		}
	});
	$effect(() => {
		if (!query || stopsMode == 'none') return;
		untrack(async () => {
			const { data } = await stops({ query });
			if (!data) {
				stopsData.length = 0;
				updateOverlayLayers(createLayer(), layers, overlay);
				return;
			}

			let index = 0;
			for (let i = 0; i < data.length; ++i) {
				metadata[index] = {
					name: data[i].name,
					stopId: data[i].stopId,
					track: data[i].track,
					modes: data[i].modes
				};
				const mode = data[i].modes ? data[i].modes![0] : undefined;
				const color = mode ? hexToRgb(getModeStyle({ mode } as LegLike)[1]) : hexToRgb('#000000');
				positions[2 * index] = data[i].lon;
				positions[2 * index + 1] = data[i].lat;
				colors[3 * index] = color[0];
				colors[3 * index + 1] = color[1];
				colors[3 * index + 2] = color[2];
				index++;
			}
			stopsData.length = index;
			updateOverlayLayers(createLayer(), layers, overlay);
		});
	});
</script>
