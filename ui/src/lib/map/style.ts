import type {
	HillshadeLayerSpecification,
	LayerSpecification,
	RasterDEMSourceSpecification,
	StyleSpecification
} from 'maplibre-gl';

// Shortbread base styles (VersaTiles Colorful / Eclipse), trimmed to `layers`
// and with their font stacks remapped to the glyphs MOTIS serves. Their source
// (`versatiles-shortbread`) is remapped to the MOTIS `osm` tile source below.
import colorful from './base/colorful.json';
import eclipse from './base/eclipse.json';

// Palette. The Shortbread base (Colorful/Eclipse) provides the outdoor colours;
// the keys below are used by the MOTIS indoor/level overlay and by other MOTIS
// map components (stops view, markers, …) that import this object.
export const colors = {
	light: {
		background: '#f8f4f0',
		water: '#99ddff',
		rail: '#a8a8a8',
		pedestrian: '#e8e7eb',
		ferryRoute: 'rgba(102, 102, 255, 0.5)',
		sport: '#d0f4be',
		sportOutline: '#b3e998',
		building: '#ded7d3',
		buildingOutline: '#cfc8c4',
		landuseComplex: '#f0e6d1',
		landuseCommercial: 'hsla(0, 60%, 87%, 0.23)',
		landuseIndustrial: '#e0e2eb',
		landuseResidential: '#ece7e4',
		landuseRetail: 'hsla(0, 60%, 87%, 0.23)',
		landuseConstruction: '#aaa69d',
		landusePark: '#b8ebad',
		landuseNatureLight: '#ddecd7',
		landuseNatureHeavy: '#a3e39c',
		landuseCemetery: '#e0e4dd',
		landuseBeach: '#fffcd3',
		indoorCorridor: '#fdfcfa',
		indoor: '#d4edff',
		indoorOutline: '#808080',
		indoorText: '#333333',
		publicTransport: 'rgba(218,140,140,0.3)',
		footway: '#7a7a7a',
		steps: '#bfbfbf',
		elevatorOutline: '#808080',
		elevator: '#2b6cb0',
		roadBackResidential: '#ffffff',
		roadBackNonResidential: '#ffffff',
		motorway: '#ffb366',
		motorwayLink: '#f7e06e',
		primarySecondary: '#fffbf8',
		linkTertiary: '#ffffff',
		residential: '#ffffff',
		road: '#ffffff',
		townText: '#333333',
		townTextHalo: 'white',
		text: '#333333',
		textHalo: 'white',
		citiesText: '#111111',
		citiesTextHalo: 'white',
		shield: 'shield'
	},
	dark: {
		background: '#292929',
		water: '#1f2830',
		rail: '#808080',
		pedestrian: '#292929',
		ferryRoute: 'rgba(58, 77, 139, 0.5)',
		sport: '#272525',
		sportOutline: '#272525',
		building: '#1F1F1F',
		buildingOutline: '#1A1A1A',
		landuseComplex: '#292929',
		landuseCommercial: '#292929',
		landuseIndustrial: '#353538',
		landuseResidential: '#292929',
		landuseRetail: '#292929',
		landuseConstruction: 'red',
		landusePark: '#18221f',
		landuseNatureLight: '#1e2322',
		landuseNatureHeavy: '#1a2020',
		landuseCemetery: '#202423',
		landuseBeach: '#4c4b3e',
		indoorCorridor: '#494949',
		indoor: '#1a1a1a',
		indoorOutline: '#0d0d0d',
		indoorText: '#eeeeee',
		publicTransport: 'rgba(89,45,45,0.405)',
		footway: '#6a6a6a',
		steps: '#70504b',
		elevatorOutline: '#808080',
		elevator: '#3b6ea5',
		roadBackResidential: '#414141',
		roadBackNonResidential: '#414141',
		motorway: '#414141',
		motorwayLink: '#414141',
		primarySecondary: '#414141',
		linkTertiary: '#414141',
		residential: '#414141',
		road: '#414141',
		text: '#9a9a9a',
		textHalo: '#151515',
		townText: '#bebebe',
		townTextHalo: '#1A1A1A',
		citiesText: '#bebebe',
		citiesTextHalo: '#1A1A1A',
		shield: 'shield-dark'
	}
};

function getUrlBase(url: string): string {
	const { origin, pathname } = new URL(url);
	return origin + pathname.slice(0, pathname.lastIndexOf('/') + 1);
}

// this doesn't escape {}-parameters
function getAbsoluteUrl(base: string, relative: string): string {
	return getUrlBase(base) + relative;
}

// The indoor / level overlay. Every layer reads from the same `osm` source as
// the base — only the `source-layer` and `filter` differ. `level` drives which
// indoor floor is shown; the whole style is rebuilt when the level changes.
function indoorLayers(level: number, c: (typeof colors)['light']): LayerSpecification[] {
	// legacy-style filters (accepted by MapLibre); typed loosely to avoid the
	// expression/legacy union friction in the type defs.
	// eslint-disable-next-line @typescript-eslint/no-explicit-any
	const atLevel: any = ['==', 'level', level];
	// eslint-disable-next-line @typescript-eslint/no-explicit-any
	const atLevelOrGround: any =
		level === 0 ? ['any', ['!has', 'level'], ['==', 'level', level]] : ['==', 'level', level];
	// eslint-disable-next-line @typescript-eslint/no-explicit-any
	const fromToLevel: any =
		level === 0
			? ['any', ['!has', 'from_level'], ['==', 'from_level', level], ['==', 'to_level', level]]
			: ['any', ['==', 'from_level', level], ['==', 'to_level', level]];

	return [
		// transit platforms (carry a level) — shortbread `land` kind=public_transport
		{
			id: 'landuse-public-transport',
			type: 'fill',
			source: 'osm',
			'source-layer': 'land',
			filter: ['all', ['==', 'kind', 'public_transport'], atLevelOrGround],
			paint: { 'fill-color': c.publicTransport }
		},
		// indoor rooms / corridors
		{
			id: 'indoor-corridor',
			type: 'fill',
			source: 'osm',
			'source-layer': 'indoor',
			minzoom: 16,
			filter: ['all', ['==', 'indoor', 'corridor'], atLevel],
			paint: { 'fill-color': c.indoorCorridor, 'fill-opacity': 0.9 }
		},
		{
			id: 'indoor',
			type: 'fill',
			source: 'osm',
			'source-layer': 'indoor',
			minzoom: 16,
			filter: ['all', ['!in', 'indoor', 'corridor', 'wall', 'elevator'], atLevel],
			paint: { 'fill-color': c.indoor, 'fill-opacity': 0.85 }
		},
		{
			id: 'indoor-outline',
			type: 'line',
			source: 'osm',
			'source-layer': 'indoor',
			minzoom: 17,
			filter: ['all', ['!in', 'indoor', 'corridor', 'wall', 'elevator'], atLevel],
			paint: { 'line-color': c.indoorOutline, 'line-width': 1.5 }
		},
		// indoor footpaths (shortbread `streets`, uses `kind` + `level`)
		{
			id: 'indoor-footway',
			type: 'line',
			source: 'osm',
			'source-layer': 'streets',
			minzoom: 16,
			filter: [
				'all',
				['in', 'kind', 'footway', 'path', 'service', 'unclassified', 'track', 'cycleway'],
				atLevelOrGround
			],
			layout: { 'line-cap': 'round' },
			paint: {
				'line-color': c.footway,
				'line-dasharray': [1, 1.5],
				'line-opacity': 0.7,
				'line-width': ['interpolate', ['linear'], ['zoom'], 16, 1, 20, 4]
			}
		},
		// stairs (shortbread `streets` kind=steps, uses from_level/to_level)
		{
			id: 'indoor-stairs',
			type: 'line',
			source: 'osm',
			'source-layer': 'streets',
			minzoom: 17,
			filter: ['all', ['==', 'kind', 'steps'], fromToLevel],
			paint: {
				'line-color': c.steps,
				'line-dasharray': [0.4, 0.4],
				'line-width': ['interpolate', ['linear'], ['zoom'], 17, 3, 20, 10]
			}
		},
		// elevators (circle + icon), visible across the levels they connect
		{
			id: 'indoor-elevator-outline',
			type: 'circle',
			source: 'osm',
			'source-layer': 'indoor',
			minzoom: 17,
			filter: [
				'all',
				['==', 'indoor', 'elevator'],
				['<=', 'from_level', level],
				['>=', 'to_level', level]
			],
			paint: { 'circle-color': c.elevatorOutline, 'circle-radius': 13 }
		},
		{
			id: 'indoor-elevator',
			type: 'circle',
			source: 'osm',
			'source-layer': 'indoor',
			minzoom: 17,
			filter: [
				'all',
				['==', 'indoor', 'elevator'],
				['<=', 'from_level', level],
				['>=', 'to_level', level]
			],
			paint: { 'circle-color': c.elevator, 'circle-radius': 11 }
		},
		{
			id: 'indoor-elevator-icon',
			type: 'symbol',
			source: 'osm',
			'source-layer': 'indoor',
			minzoom: 17,
			filter: [
				'all',
				['==', 'indoor', 'elevator'],
				['<=', 'from_level', level],
				['>=', 'to_level', level]
			],
			layout: { 'icon-image': 'elevator', 'icon-size': 0.85 }
		},
		// indoor room labels
		{
			id: 'indoor-names',
			type: 'symbol',
			source: 'osm',
			'source-layer': 'indoor',
			minzoom: 18,
			filter: ['any', ['!has', 'level'], ['==', 'level', level]],
			layout: {
				'symbol-placement': 'point',
				'text-field': ['get', 'name'],
				'text-font': ['Noto Sans Regular'],
				'text-size': 11
			},
			paint: { 'text-color': c.indoorText, 'text-halo-color': '#ffffff', 'text-halo-width': 1 }
		}
	];
}

export const getStyle = (
	theme: 'light' | 'dark',
	level: number,
	staticBaseUrl: string,
	apiBaseUrl: string,
	withHillshades: boolean
): StyleSpecification => {
	const c = colors[theme];

	// Shortbread base (light = Colorful, dark = Eclipse); point every base layer
	// at the MOTIS `osm` tile source.
	const base = (theme === 'dark' ? eclipse : colorful) as unknown as {
		layers: LayerSpecification[];
	};
	const baseLayers = base.layers.map((l) =>
		'source' in l && l.source ? ({ ...l, source: 'osm' } as LayerSpecification) : l
	);

	// MOTIS fills the sea from the coastline shapefile into a `coastline` layer
	// (the Shortbread base's `ocean` layer stays empty). Draw it right above the
	// base background so land keeps the base colour and the sea is water-coloured.
	baseLayers.splice(1, 0, {
		id: 'coastline',
		type: 'fill',
		source: 'osm',
		'source-layer': 'coastline',
		paint: { 'fill-color': c.water }
	} as LayerSpecification);

	const hillshadeSources: StyleSpecification['sources'] = withHillshades
		? {
			hillshadeSource: {
				type: 'raster-dem',
				tiles: ['./mapterhorn/{z}/{x}/{y}.webp'],
				attribution: "<a href='https://mapterhorn.com/attribution'>© Mapterhorn</a>",
				bounds: [-180, -85.0511287, 180, 85.0511287],
				encoding: 'terrarium',
				tileSize: 512
			} satisfies RasterDEMSourceSpecification
		}
		: {};
	const hillshadeLayers: HillshadeLayerSpecification[] = withHillshades
		? [
			{
				id: 'hillshade',
				type: 'hillshade',
				source: 'hillshadeSource',
				paint: { 'hillshade-exaggeration': 0.33 }
			}
		]
		: [];

	// Insert hillshade above the base land/water fills but below streets.
	const streetIdx = baseLayers.findIndex(
		(l) => typeof l.id === 'string' && (l.id.startsWith('street') || l.id.startsWith('bridge'))
	);
	const baseWithHillshade =
		streetIdx >= 0
			? [...baseLayers.slice(0, streetIdx), ...hillshadeLayers, ...baseLayers.slice(streetIdx)]
			: [...baseLayers, ...hillshadeLayers];

	return {
		version: 8,
		sources: {
			osm: {
				type: 'vector',
				tiles: [getAbsoluteUrl(apiBaseUrl, 'tiles/{z}/{x}/{y}.mvt')],
				maxzoom: 20,
				attribution: '© OpenStreetMap contributors'
			},
			...hillshadeSources
		},
		glyphs: getAbsoluteUrl(staticBaseUrl, 'glyphs/{fontstack}/{range}.pbf'),
		// Two sprites: the base's POI icons are referenced as "basics:…"; MOTIS's
		// own icons (e.g. the elevator) come from the default `sprite_sdf`.
		sprite: [
			{ id: 'default', url: getAbsoluteUrl(staticBaseUrl, 'sprite_sdf') },
			{ id: 'basics', url: getAbsoluteUrl(staticBaseUrl, 'sprites/basics/sprites') }
		],
		layers: [...baseWithHillshade, ...indoorLayers(level, c)]
	};
};
